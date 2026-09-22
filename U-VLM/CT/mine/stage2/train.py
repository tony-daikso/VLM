"""
Stage 2 訓練 loop：兩階段 encoder fine-tune（前 `freeze_encoder_epochs` epoch 凍結 encoder
只練 head，之後解凍、用 discriminative LR 一起微調，見 STAGE2_PLAN.md 第 5、8 節）。
Early stopping / best checkpoint 都以 val macro AUROC 為準（只對 val 裡兩個 class 都有出現的
標籤取平均，見 compute_label_metrics）。

Checkpoint：
  best_encoder.pt   —— 微調後的 encoder state_dict，給 Stage 3 用
  best_head.pt      —— 分類 head state_dict，自己 eval 用
  last.pt           —— 最新一個 epoch 的完整模型 + optimizer/scheduler state，方便中斷後恢復
                        （遠端長時間訓練容易因為睡眠/斷線中斷，見 HANDOFF.md 第 4 節的教訓）

跑法：python3 U-VLM/stage2/train.py [--config U-VLM/stage2/config.yaml]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import Stage2Dataset
from losses import build_criterion, compute_pos_weight
from model import Stage2Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 專案根目錄


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


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def move_batch_to_device(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["label"] = batch["label"].to(device)
    return batch


def compute_label_metrics(probs, labels, label_names):
    """probs/labels: (N, num_labels) numpy。標籤只有單一 class（val 裡全 0 或全 1）時
    AUROC/AUPRC 沒有意義，設成 None，不硬算（sklearn 對這種情況會直接報錯）。"""
    result = {}
    for i, name in enumerate(label_names):
        y = labels[:, i]
        p = probs[:, i]
        n_pos = int(y.sum())
        if 0 < n_pos < len(y):
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
        else:
            auroc, auprc = None, None
        result[name] = {"n_pos": n_pos, "auroc": auroc, "auprc": auprc}
    return result


def macro_mean(label_metrics, key):
    values = [m[key] for m in label_metrics.values() if m[key] is not None]
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def run_validation(model, val_loader, label_names, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["image"])
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    label_metrics = compute_label_metrics(probs, labels, label_names)

    return {
        "label_metrics": label_metrics,
        "macro_auroc": macro_mean(label_metrics, "auroc"),
        "macro_auprc": macro_mean(label_metrics, "auprc"),
    }


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
    parser.add_argument("--resume", default=None, help="從 last.pt 之類的 checkpoint 恢復訓練")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    print(f"device: {device}")

    label_names = config["labels"]
    train_ds = Stage2Dataset(config, "train", augment=True)
    val_ds = Stage2Dataset(config, "val", augment=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_cfg = config["train"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )

    model = Stage2Model(config).to(device)
    pos_weight = compute_pos_weight(train_ds, label_names, train_cfg["pos_weight_cap"]).to(device)
    criterion = build_criterion(pos_weight)
    print("pos_weight per label:", {n: round(w, 2) for n, w in zip(label_names, pos_weight.cpu().tolist())})

    optimizer = build_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    checkpoint_dir = resolve_path(train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    encoder_checkpoint = resolve_path(config["data"]["encoder_checkpoint"])

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
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            logits = model(batch["image"])
            loss = criterion(logits, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % train_cfg["log_every_n_steps"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader)

        val_metrics = run_validation(model, val_loader, label_names, device)
        score = val_metrics["macro_auroc"]

        elapsed = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_macro_auroc={val_metrics['macro_auroc']:.4f} "
              f"val_macro_auprc={val_metrics['macro_auprc']:.4f} "
              f"({'frozen' if frozen else 'finetuning'}, {elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "frozen": frozen,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_label_metrics": val_metrics["label_metrics"],
            "score": score,
        })
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
            torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, "best_encoder.pt"))
            torch.save(model.head.state_dict(), os.path.join(checkpoint_dir, "best_head.pt"))
            print(f"  -> new best macro AUROC {best_score:.4f}, saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs)")
                break


if __name__ == "__main__":
    main()
