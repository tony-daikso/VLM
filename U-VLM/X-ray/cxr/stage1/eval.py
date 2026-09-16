"""
跑 val（或 test）set，輸出 lesion head 的 Dice 分布（見 STAGE1_PLAN.md 第 6 節）：
  - per-sample Dice 分布（mean/median/min/max + 存成 csv）
  - 幾張 overlay png（image + GT box-mask vs 預測），方便肉眼檢查

跑法：python3 U-VLM/X-ray/cxr/stage1/eval.py --checkpoint U-VLM/X-ray/cxr/stage1/checkpoints/best_full.pt
"""
import argparse
import csv
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset import Stage1Dataset
from model import Stage1Model
from train import get_device, move_batch_to_device, resolve_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def collect_metrics(model, loader, device):
    dices, ids = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        probs = (torch.sigmoid(outputs["lesion_logits"]) > 0.5).float()
        target = batch["lesion_mask"]
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist()
        dices += dice
        ids += list(batch["study_id"])
    return {"ids": ids, "dices": dices}


def summarize(name, dices):
    if not dices:
        print(f"{name}: no samples")
        return
    arr = np.array(dices)
    print(
        f"{name}: n={len(arr)} mean={arr.mean():.4f} median={np.median(arr):.4f} "
        f"min={arr.min():.4f} max={arr.max():.4f}"
    )


def write_dice_csv(path, ids, dices):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "dice"])
        for study_id, dice in zip(ids, dices):
            writer.writerow([study_id, dice])


@torch.no_grad()
def save_qa_overlays(model, ds, device, out_dir, num_overlays):
    os.makedirs(out_dir, exist_ok=True)
    for idx in range(min(num_overlays, len(ds))):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        outputs = model(image)

        img_np = ((sample["image"][0].numpy() + 1) / 2 * 255).astype(np.uint8)
        gt = (sample["lesion_mask"][0].numpy() * 255).astype(np.uint8)
        pred = (torch.sigmoid(outputs["lesion_logits"][0, 0]).cpu().numpy() > 0.5).astype(np.uint8) * 255

        panels = [Image.fromarray(a).convert("RGB") for a in (img_np, gt, pred)]
        combined = Image.new("RGB", (img_np.shape[1] * len(panels), img_np.shape[0]))
        for i, panel in enumerate(panels):
            combined.paste(panel, (i * img_np.shape[1], 0))
        combined.save(os.path.join(out_dir, f"{sample['study_id']}_qa.png"))

    print(f"saved {min(num_overlays, len(ds))} QA overlays to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    model = Stage1Model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    ds = Stage1Dataset(config, args.split, augment=False)
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False,
                         num_workers=config["train"]["num_workers"])

    metrics = collect_metrics(model, loader, device)

    print(f"\n=== {args.split} Dice ===")
    summarize("lesion", metrics["dices"])

    checkpoint_dir = os.path.dirname(args.checkpoint)
    write_dice_csv(os.path.join(checkpoint_dir, f"eval_lesion_dice_{args.split}.csv"),
                    metrics["ids"], metrics["dices"])

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
    qa_dir = resolve_path(root_dir, config["eval"]["qa_overlay_dir"])
    save_qa_overlays(model, ds, device, qa_dir, config["eval"]["num_qa_overlays"])


if __name__ == "__main__":
    main()
