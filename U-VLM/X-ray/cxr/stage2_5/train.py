"""
Stage 2.5 訓練 loop：兩階段 encoder fine-tune（前 `freeze_encoder_epochs` epoch 凍結 encoder
只練 head，之後解凍、用 discriminative LR 一起微調，跟 Stage 2 一致，見 STAGE2_5_PLAN.md 第 5 節）。
Best checkpoint 以 val macro mAP 為準（見 STAGE2_5_PLAN.md 第 6 節，IoU 門檻 0.3）。

Checkpoint：
  best_encoder.pt / best_head.pt / last.pt

跑法：python3 U-VLM/X-ray/cxr/stage2_5/train.py [--config .../config.yaml] [--epochs N] [--resume last.pt]
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import Stage2_5Dataset, collect_boxes_and_classes
from decode import compute_ap_for_class, decode_predictions
from losses import compute_losses
from model import Stage2_5Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(root_dir, path):
    return path if os.path.isabs(path) else os.path.join(root_dir, path)


def move_batch_to_device(batch, device):
    for key in ["image", "heatmap", "size_target", "offset_target", "reg_mask"]:
        batch[key] = batch[key].to(device)
    return batch


def build_gt_boxes_by_class(dataset, label_names):
    """{class_idx: {study_id: [box, ...]}}，val/test augment=False，座標就是 manifest 原始 box，
    不用重跑一次 dataset 的 augmentation 路徑。"""
    gt = {i: defaultdict(list) for i in range(len(label_names))}
    for row in dataset.records:
        boxes, class_index_lists = collect_boxes_and_classes(row, dataset.label_to_group, dataset.label_index)
        for box, classes in zip(boxes, class_index_lists):
            for cls in classes:
                gt[cls][row["study_id"]].append(box)
    return gt


@torch.no_grad()
def run_validation(model, val_loader, gt_boxes_by_class, label_names, device, iou_threshold, topk):
    model.eval()
    pred_boxes_by_class = {i: defaultdict(list) for i in range(len(label_names))}

    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        dets_per_sample = decode_predictions(outputs["heatmap"], outputs["size"], outputs["offset"], topk)
        for study_id, dets in zip(batch["study_id"], dets_per_sample):
            for score, cls, x1, y1, x2, y2 in dets:
                pred_boxes_by_class[cls][study_id].append((score, (x1, y1, x2, y2)))

    aps, n_gts = [], []
    per_label = {}
    for i, name in enumerate(label_names):
        ap, n_gt = compute_ap_for_class(gt_boxes_by_class[i], pred_boxes_by_class[i], iou_threshold)
        per_label[name] = {"ap": ap, "n_gt": n_gt}
        if ap is not None:
            aps.append(ap)
        n_gts.append(n_gt)

    macro_map = float(np.mean(aps)) if aps else 0.0
    return {"macro_map": macro_map, "per_label": per_label}


def build_optimizer(model, train_cfg):
    return torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": train_cfg["head_lr"]},
            {"params": model.encoder.parameters(), "lr": train_cfg["encoder_lr"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    set_seed(config["seed"])
    device = get_device()
    print(f"device: {device}")

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
    label_names = config["labels"]
    train_ds = Stage2_5Dataset(config, "train", augment=True)
    val_ds = Stage2_5Dataset(config, "validation", augment=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_cfg = config["train"]
    eval_cfg = config["eval"]
    persistent_workers = train_cfg["num_workers"] > 0
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], drop_last=len(train_ds) > train_cfg["batch_size"],
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"],
        persistent_workers=persistent_workers,
    )
    gt_boxes_by_class = build_gt_boxes_by_class(val_ds, label_names)

    model = Stage2_5Model(config).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    checkpoint_dir = resolve_path(root_dir, train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    encoder_checkpoint = resolve_path(root_dir, config["data"]["encoder_checkpoint"])

    start_epoch = 0
    best_score = -1.0
    epochs_without_improvement = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")
    else:
        state_dict = torch.load(encoder_checkpoint, map_location=device)
        model.encoder.load_state_dict(state_dict)
        print(f"loaded Stage 1 encoder weights from {encoder_checkpoint}")

    freeze_epochs = train_cfg["freeze_encoder_epochs"]
    history = []
    for epoch in range(start_epoch, train_cfg["epochs"]):
        frozen = epoch < freeze_epochs
        if epoch == freeze_epochs:
            print(f"[epoch {epoch}] unfreezing encoder, switching to discriminative LR fine-tune")

        model.train()
        model.set_encoder_frozen(frozen)
        t0 = time.time()
        running = {"total": 0.0, "heatmap": 0.0, "size": 0.0, "offset": 0.0}

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            losses = compute_losses(outputs, batch, config)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()

            if step % train_cfg["log_every_n_steps"] == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} total={losses['total'].item():.4f} "
                    f"heatmap={losses['heatmap'].item():.4f} size={losses['size'].item():.4f} "
                    f"offset={losses['offset'].item():.4f}"
                )

        scheduler.step()
        n_steps = max(len(train_loader), 1)
        train_summary = {k: v / n_steps for k, v in running.items()}

        val_metrics = run_validation(
            model, val_loader, gt_boxes_by_class, label_names, device,
            eval_cfg["iou_threshold"], eval_cfg["topk"],
        )
        score = val_metrics["macro_map"]

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={train_summary['total']:.4f} "
            f"val_macro_map={score:.4f} ({'frozen' if frozen else 'finetuning'}, {elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch, "train": train_summary, "frozen": frozen,
            "val_macro_map": score, "val_per_label": val_metrics["per_label"], "score": score,
        })
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            is_new_best = True
        else:
            epochs_without_improvement += 1
            is_new_best = False

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": config,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "last.pt"))

        if is_new_best:
            torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, "best_encoder.pt"))
            torch.save(model.head.state_dict(), os.path.join(checkpoint_dir, "best_head.pt"))
            print(f"  -> new best macro mAP {best_score:.4f}, saved checkpoint")
        else:
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {train_cfg['early_stopping_patience']} epochs)"
                )
                break


if __name__ == "__main__":
    main()
