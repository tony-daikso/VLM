"""
Stage 1 訓練 loop：混合抽樣 CT-RATE(3,042) + HiPaS(250) 兩個來源
（WeightedRandomSampler 讓兩者在一個 epoch 內出現次數大致平衡，見 STAGE1_PLAN.md 第 2、5 節），
每個 epoch 結束後跑 val，monitor 三個 head 各自的 val Dice，early stopping 看三者是否都沒有進步。

Checkpoint：
  best_encoder.pt   —— 只存 encoder state_dict，給 Stage 2/3 用
  best_full.pt      —— 完整三頭模型，給這階段自己 eval 用
  last.pt           —— 最新一個 epoch 的完整模型 + optimizer state，方便中斷後恢復

跑法：python3 U-VLM/stage1/train.py [--config U-VLM/stage1/config.yaml]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def make_weighted_sampler(dataset, hipas_ratio):
    sources = dataset.source_flags()
    n_ctrate = sources.count("ctrate")
    n_hipas = sources.count("hipas")
    # 讓 hipas 樣本被抽到的總權重 ≈ hipas_ratio * ctrate 樣本的總權重，
    # 兩來源在一個 epoch 內出現次數大致平衡（而不是照實際筆數 12:1 的比例）
    weight_ctrate = 1.0
    weight_hipas = hipas_ratio * n_ctrate / max(n_hipas, 1)
    weights = [weight_ctrate if s == "ctrate" else weight_hipas for s in sources]
    return WeightedRandomSampler(weights, num_samples=len(sources), replacement=True)


@torch.no_grad()
def dice_per_sample_binary(logits, target, valid_mask, threshold=0.5):
    if valid_mask.sum() == 0:
        return []
    probs = (torch.sigmoid(logits[valid_mask]) > threshold).float()
    target = target[valid_mask]
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.cpu().tolist()


@torch.no_grad()
def dice_per_class_multiclass(logits, target, valid_mask, num_classes):
    """回傳 (num_classes,) tensor，每個 class 在這個 batch valid 樣本上的 dice 累積
    intersection/union（跨 batch 累積後再算 dice，避免類別在單一 batch 沒出現時的雜訊）。"""
    if valid_mask.sum() == 0:
        return torch.zeros(num_classes), torch.zeros(num_classes)
    pred = torch.argmax(logits[valid_mask], dim=1)
    target = target[valid_mask]
    pred_onehot = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (pred_onehot * target_onehot).sum(dim=dims)
    union = pred_onehot.sum(dim=dims) + target_onehot.sum(dim=dims)
    return intersection.cpu(), union.cpu()


def move_batch_to_device(batch, device):
    for key in ["image", "lesion_mask", "organ_mask", "vessel_mask", "valid_lesion_organ", "valid_vessel"]:
        batch[key] = batch[key].to(device)
    return batch


def run_validation(model, val_loader, config, device):
    model.eval()
    num_organ_classes = config["organ"]["num_classes"]

    lesion_dices, vessel_artery_dices, vessel_vein_dices = [], [], []
    organ_intersection = torch.zeros(num_organ_classes)
    organ_union = torch.zeros(num_organ_classes)

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])

            lesion_dices += dice_per_sample_binary(
                outputs["lesion_logits"], batch["lesion_mask"], batch["valid_lesion_organ"]
            )
            vessel_artery_dices += dice_per_sample_binary(
                outputs["vessel_logits"][:, 0:1], batch["vessel_mask"][:, 0:1], batch["valid_vessel"]
            )
            vessel_vein_dices += dice_per_sample_binary(
                outputs["vessel_logits"][:, 1:2], batch["vessel_mask"][:, 1:2], batch["valid_vessel"]
            )
            inter, uni = dice_per_class_multiclass(
                outputs["organ_logits"], batch["organ_mask"], batch["valid_lesion_organ"], num_organ_classes
            )
            organ_intersection += inter
            organ_union += uni

    organ_per_class_dice = (2 * organ_intersection + 1e-6) / (organ_union + 1e-6)
    # 只對這個 val split 裡有出現過（union>0）的 class 取 macro mean，
    # 沒出現過的 class 不該影響 mean（不是模型學不好，是 val 裡根本沒有這個結構）
    present = organ_union > 0
    organ_macro_dice = organ_per_class_dice[present].mean().item() if present.any() else 0.0

    return {
        "lesion_dice": float(np.mean(lesion_dices)) if lesion_dices else 0.0,
        "organ_macro_dice": organ_macro_dice,
        "vessel_artery_dice": float(np.mean(vessel_artery_dices)) if vessel_artery_dices else 0.0,
        "vessel_vein_dice": float(np.mean(vessel_vein_dices)) if vessel_vein_dices else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--resume", default=None, help="從 last.pt 之類的 checkpoint 恢復訓練")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    print(f"device: {device}")

    train_ds = Stage1Dataset(config, "train", augment=True)
    val_ds = Stage1Dataset(config, "val", augment=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_cfg = config["train"]
    sampler = make_weighted_sampler(train_ds, train_cfg["hipas_sample_weight_ratio"])
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
        num_workers=train_cfg["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )

    model = Stage1Model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    root_dir = os.path.dirname(SCRIPT_DIR)  # U-VLM/
    root_dir = os.path.dirname(root_dir)  # 專案根目錄
    checkpoint_dir = train_cfg["checkpoint_dir"] if os.path.isabs(train_cfg["checkpoint_dir"]) \
        else os.path.join(root_dir, train_cfg["checkpoint_dir"])
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
        running = {"total": 0.0, "lesion": 0.0, "organ": 0.0, "vessel": 0.0}

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
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"total={losses['total'].item():.4f} lesion={losses['lesion'].item():.4f} "
                      f"organ={losses['organ'].item():.4f} vessel={losses['vessel'].item():.4f}")

        scheduler.step()
        n_steps = len(train_loader)
        train_summary = {k: v / n_steps for k, v in running.items()}

        val_metrics = run_validation(model, val_loader, config, device)
        # 綜合分數：三個 head 的 val dice 平均，用來決定 best checkpoint / early stopping，
        # 但三個 head 各自的數字都會存進 history，不會只看這一個數字就下結論
        score = (val_metrics["lesion_dice"] + val_metrics["organ_macro_dice"]
                 + (val_metrics["vessel_artery_dice"] + val_metrics["vessel_vein_dice"]) / 2) / 3

        elapsed = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_summary['total']:.4f} "
              f"val_lesion_dice={val_metrics['lesion_dice']:.4f} "
              f"val_organ_macro_dice={val_metrics['organ_macro_dice']:.4f} "
              f"val_vessel_artery_dice={val_metrics['vessel_artery_dice']:.4f} "
              f"val_vessel_vein_dice={val_metrics['vessel_vein_dice']:.4f} "
              f"score={score:.4f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train": train_summary, "val": val_metrics, "score": score})
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": config,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "last.pt"))

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(model.encoder_state_dict(), os.path.join(checkpoint_dir, "best_encoder.pt"))
            torch.save(ckpt, os.path.join(checkpoint_dir, "best_full.pt"))
            print(f"  -> new best score {best_score:.4f}, saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs)")
                break


if __name__ == "__main__":
    main()
