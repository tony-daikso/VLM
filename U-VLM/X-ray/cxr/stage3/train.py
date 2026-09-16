"""
Stage 3 訓練 loop：teacher forcing，encoder 凍結（見 STAGE3_PLAN.md 第 6 節），只訓練
visual_tokenizer 的投影/位置編碼 + GPT-2 全部參數（包含隨機初始化的 cross-attention）。
Best checkpoint 以 val loss（negative log-likelihood）為準，越低越好。

Checkpoint：
  best_model.pt   -- 完整模型（encoder 權重也在裡面，方便 eval.py 直接載入，不用再額外指定
                      Stage 2 encoder checkpoint）
  last.pt         -- 最新一個 epoch，方便中斷後恢復

跑法：python3 U-VLM/X-ray/cxr/stage3/train.py [--config .../config.yaml] [--epochs N] [--resume last.pt]
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
from transformers import GPT2Tokenizer

from dataset import Stage3Dataset, make_collate_fn
from model import Stage3Model

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
    for key in ["image", "input_ids", "attention_mask", "labels"]:
        batch[key] = batch[key].to(device)
    return batch


def build_tokenizer(lm_name):
    tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"], batch["input_ids"], batch["attention_mask"], batch["labels"])
        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


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
    tokenizer = build_tokenizer(config["model"]["lm_name"])

    train_ds = Stage3Dataset(config, "train", tokenizer)
    val_ds = Stage3Dataset(config, "validation", tokenizer)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_cfg = config["train"]
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], drop_last=len(train_ds) > train_cfg["batch_size"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    model = Stage3Model(config).to(device)
    encoder_checkpoint = resolve_path(root_dir, config["data"]["encoder_checkpoint"])

    checkpoint_dir = resolve_path(root_dir, train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_score = float("inf")
    epochs_without_improvement = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", float("inf"))
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")
    else:
        state_dict = torch.load(encoder_checkpoint, map_location=device)
        model.encoder.load_state_dict(state_dict)
        print(f"loaded Stage 2 encoder weights from {encoder_checkpoint}")

    history = []
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"], batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % train_cfg["log_every_n_steps"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = run_validation(model, val_loader, device)
        score = val_loss  # 越低越好

        elapsed = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if score < best_score:
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
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  -> new best val_loss {best_score:.4f}, saved checkpoint")
        else:
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {train_cfg['early_stopping_patience']} epochs)"
                )
                break


if __name__ == "__main__":
    main()
