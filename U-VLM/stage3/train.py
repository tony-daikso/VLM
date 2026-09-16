"""
Stage 3 訓練 loop：encoder 全程凍結（Table 5 ablation 顯示凍結比繼續微調好，見
STAGE3_PLAN.md 第 2.3 節），只訓練 decoder（5 層 multi-layer injection Transformer +
per-stage projection）。

Checkpoint：
  best_decoder.pt   —— val loss 最低時的 decoder state_dict（含 aligner/projection 層）
  last.pt           —— 最新一個 epoch 的完整模型 + optimizer/scheduler state，方便中斷後恢復
  tokenizer/         —— 借用的 GPT-2 tokenizer 檔案，存一份方便 eval.py / 之後推論直接載入

跑法：python3 U-VLM/stage3/train.py [--config U-VLM/stage3/config.yaml]
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

from dataset import Stage3Dataset, build_tokenizer
from losses import report_generation_loss
from model import Stage3Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


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
    for key in ("image", "instruction_ids", "report_ids", "report_mask"):
        batch[key] = batch[key].to(device)
    return batch


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["image"], batch["instruction_ids"], batch["report_ids"], batch["report_mask"])
        loss = report_generation_loss(logits, batch["report_ids"], batch["report_mask"])
        total_loss += loss.item() * batch["image"].shape[0]
        total_count += batch["image"].shape[0]
    return total_loss / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    print(f"device: {device}")

    tokenizer = build_tokenizer(config["text"]["tokenizer_name"])
    train_ds = Stage3Dataset(config, "train", augment=True, tokenizer=tokenizer)
    val_ds = Stage3Dataset(config, "val", augment=False, tokenizer=tokenizer)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples, vocab: {tokenizer.vocab_size}")

    train_cfg = config["train"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )

    model = Stage3Model(config, vocab_size=tokenizer.vocab_size).to(device)

    encoder_checkpoint = resolve_path(config["data"]["encoder_checkpoint"])
    state_dict = torch.load(encoder_checkpoint, map_location=device)
    model.encoder.load_state_dict(state_dict)
    for p in model.encoder.parameters():
        p.requires_grad = False
    print(f"loaded Stage 2 encoder weights from {encoder_checkpoint} (frozen)")

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=train_cfg["decoder_lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    checkpoint_dir = resolve_path(train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))

    start_epoch = 0
    best_score = float("inf")
    epochs_without_improvement = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", float("inf"))
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        model.encoder.eval()  # 凍結的 encoder 永遠是 eval 模式（BN 統計量不能被訓練 batch 影響）
        t0 = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            logits = model(batch["image"], batch["instruction_ids"], batch["report_ids"], batch["report_mask"])
            loss = report_generation_loss(logits, batch["report_ids"], batch["report_mask"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % train_cfg["log_every_n_steps"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss = run_validation(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
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

        if val_loss < best_score:
            best_score = val_loss
            epochs_without_improvement = 0
            torch.save(model.decoder.state_dict(), os.path.join(checkpoint_dir, "best_decoder.pt"))
            print(f"  -> new best val_loss {best_score:.4f}, saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs)")
                break


if __name__ == "__main__":
    main()
