"""
Stage 1 訓練 loop（單一 lesion head，單一資料來源，不需要 CT 版的 WeightedRandomSampler）。
每個 epoch 結束後跑 val，monitor val Dice，early stopping。

Checkpoint：
  best_encoder.pt   -- 只存 encoder state_dict，給 Stage 2/3 用
  best_full.pt      -- 完整模型，給這階段自己 eval 用
  last.pt           -- 最新一個 epoch 的完整模型 + optimizer state，方便中斷後恢復

跑法：python3 U-VLM/X-ray/cxr/stage1/train.py [--config .../config.yaml] [--epochs N]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import Stage1Dataset
from losses import compute_losses
from model import Stage1Model

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


def move_batch_to_device(batch, device):
    for key in ["image", "lesion_mask"]:
        batch[key] = batch[key].to(device)
    return batch


@torch.no_grad()
def dice_per_sample(logits, target, threshold=0.5):
    probs = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.cpu().tolist()


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    dices = []
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        dices += dice_per_sample(outputs["lesion_logits"], batch["lesion_mask"])
    return {"lesion_dice": float(np.mean(dices)) if dices else 0.0}


def resolve_path(root_dir, path):
    return path if os.path.isabs(path) else os.path.join(root_dir, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--resume", default=None, help="從 last.pt 之類的 checkpoint 恢復訓練")
    parser.add_argument("--epochs", type=int, default=None, help="覆寫 config 的 epochs（quick smoke test 用）")
    parser.add_argument("--limit", type=int, default=None, help="train/val 各只取前 N 筆（quick smoke test 用）")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.limit is not None:
        config["train"]["num_workers"] = 0  # 資料量太小，起 worker process 的 overhead 反而更慢
        config["train"]["batch_size"] = min(config["train"]["batch_size"], args.limit)

    set_seed(config["seed"])
    device = get_device()
    print(f"device: {device}")

    train_ds = Stage1Dataset(config, "train", augment=True, limit=args.limit)
    val_ds = Stage1Dataset(config, "validation", augment=False, limit=args.limit)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_cfg = config["train"]
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

    model = Stage1Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
    checkpoint_dir = resolve_path(root_dir, train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    history = []
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "dice": 0.0, "bce": 0.0}

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
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"total={losses['total'].item():.4f} dice={losses['dice'].item():.4f} "
                    f"bce={losses['bce'].item():.4f}"
                )

        scheduler.step()
        n_steps = max(len(train_loader), 1)
        train_summary = {k: v / n_steps for k, v in running.items()}

        val_metrics = run_validation(model, val_loader, device)
        score = val_metrics["lesion_dice"]

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={train_summary['total']:.4f} "
            f"val_lesion_dice={score:.4f} ({elapsed:.1f}s)"
        )

        history.append({"epoch": epoch, "train": train_summary, "val": val_metrics, "score": score})
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
            torch.save(model.encoder_state_dict(), os.path.join(checkpoint_dir, "best_encoder.pt"))
            torch.save(ckpt, os.path.join(checkpoint_dir, "best_full.pt"))
            print(f"  -> new best score {best_score:.4f}, saved checkpoint")
        else:
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {train_cfg['early_stopping_patience']} epochs)"
                )
                break


if __name__ == "__main__":
    main()
