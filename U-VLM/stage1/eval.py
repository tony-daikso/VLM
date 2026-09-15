"""
跑 val set，輸出三個 head 的 Dice 分布（見 STAGE1_PLAN.md 第 6 節）：
  - lesion/vessel(artery/vein)：per-sample Dice 分布（mean/median/min/max + 存成 csv）
  - organ：per-class Dice（只列 val 裡有出現過的 class）+ macro mean，另外挑幾個常見結構
    （肺葉/心臟/主動脈等）額外列出，避免稀有結構拉低 macro mean 誤導判斷
  - 額外存幾張 overlay png（image + GT vs 預測），方便肉眼檢查

跑法：python3 U-VLM/stage1/eval.py --checkpoint U-VLM/stage1/checkpoints/best_full.pt
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset import Stage1Dataset
from model import Stage1Model
from train import get_device, move_batch_to_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# TotalSegmentator "total" task 的常見大結構（id 對照見
# totalsegmentator.map_to_binary.class_map["total"]），organ per-class 報告時額外列出，
# 因為稀有結構（如某幾根肋骨）樣本少、macro mean 容易被拉低但不代表模型真的學不好
NOTABLE_ORGAN_CLASSES = {
    5: "liver", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    26: "heart", 51: "aorta", 42: "esophagus", 44: "trachea",
}


@torch.no_grad()
def collect_metrics(model, loader, config, device):
    num_organ_classes = config["organ"]["num_classes"]

    lesion_dices, artery_dices, vein_dices = [], [], []
    lesion_ids, artery_ids, vein_ids = [], [], []
    organ_intersection = torch.zeros(num_organ_classes)
    organ_union = torch.zeros(num_organ_classes)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])

        valid_lo = batch["valid_lesion_organ"]
        valid_v = batch["valid_vessel"]

        if valid_lo.any():
            probs = (torch.sigmoid(outputs["lesion_logits"][valid_lo]) > 0.5).float()
            target = batch["lesion_mask"][valid_lo]
            dims = (1, 2, 3)
            inter = (probs * target).sum(dim=dims)
            union = probs.sum(dim=dims) + target.sum(dim=dims)
            dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist()
            lesion_dices += dice
            lesion_ids += [i for i, v in zip(batch["item_id"], valid_lo.cpu().tolist()) if v]

            pred = torch.argmax(outputs["organ_logits"][valid_lo], dim=1)
            tgt = batch["organ_mask"][valid_lo]
            pred_oh = F.one_hot(pred, num_organ_classes).permute(0, 3, 1, 2).float()
            tgt_oh = F.one_hot(tgt, num_organ_classes).permute(0, 3, 1, 2).float()
            organ_intersection += (pred_oh * tgt_oh).sum(dim=(0, 2, 3)).cpu()
            organ_union += (pred_oh.sum(dim=(0, 2, 3)) + tgt_oh.sum(dim=(0, 2, 3))).cpu()

        if valid_v.any():
            for ch, dices, ids in [(0, artery_dices, artery_ids), (1, vein_dices, vein_ids)]:
                probs = (torch.sigmoid(outputs["vessel_logits"][valid_v, ch:ch + 1]) > 0.5).float()
                target = batch["vessel_mask"][valid_v, ch:ch + 1]
                dims = (1, 2, 3)
                inter = (probs * target).sum(dim=dims)
                union = probs.sum(dim=dims) + target.sum(dim=dims)
                dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist()
                dices += dice
                ids += [i for i, v in zip(batch["item_id"], valid_v.cpu().tolist()) if v]

    organ_per_class_dice = (2 * organ_intersection + 1e-6) / (organ_union + 1e-6)
    present = organ_union > 0

    return {
        "lesion": {"ids": lesion_ids, "dices": lesion_dices},
        "vessel_artery": {"ids": artery_ids, "dices": artery_dices},
        "vessel_vein": {"ids": vein_ids, "dices": vein_dices},
        "organ_per_class_dice": organ_per_class_dice,
        "organ_class_present": present,
    }


def summarize(name, dices):
    if not dices:
        print(f"{name}: no samples")
        return
    arr = np.array(dices)
    print(f"{name}: n={len(arr)} mean={arr.mean():.4f} median={np.median(arr):.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")


def write_dice_csv(path, ids, dices):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "dice"])
        for item_id, dice in zip(ids, dices):
            writer.writerow([item_id, dice])


@torch.no_grad()
def save_qa_overlays(model, val_ds, config, device, out_dir, num_overlays):
    os.makedirs(out_dir, exist_ok=True)
    ctrate_indices = [i for i, (s, _) in enumerate(val_ds.items) if s == "ctrate"][:num_overlays // 2]
    hipas_indices = [i for i, (s, _) in enumerate(val_ds.items) if s == "hipas"][:num_overlays // 2]

    for idx in ctrate_indices + hipas_indices:
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        outputs = model(image)

        img_np = ((sample["image"][0].numpy() + 1) / 2 * 255).astype(np.uint8)
        panels = [Image.fromarray(img_np).convert("RGB")]

        if sample["valid_lesion_organ"]:
            gt = (sample["lesion_mask"][0].numpy() * 255).astype(np.uint8)
            pred = (torch.sigmoid(outputs["lesion_logits"][0, 0]).cpu().numpy() > 0.5).astype(np.uint8) * 255
            panels.append(Image.fromarray(gt).convert("RGB"))
            panels.append(Image.fromarray(pred).convert("RGB"))
        else:
            artery_gt = (sample["vessel_mask"][0].numpy() * 255).astype(np.uint8)
            artery_pred = (torch.sigmoid(outputs["vessel_logits"][0, 0]).cpu().numpy() > 0.5).astype(np.uint8) * 255
            panels.append(Image.fromarray(artery_gt).convert("RGB"))
            panels.append(Image.fromarray(artery_pred).convert("RGB"))

        combined = Image.new("RGB", (img_np.shape[1] * len(panels), img_np.shape[0]))
        for i, panel in enumerate(panels):
            combined.paste(panel, (i * img_np.shape[1], 0))
        combined.save(os.path.join(out_dir, f"{sample['item_id']}_{sample['source']}.png"))

    print(f"saved {len(ctrate_indices) + len(hipas_indices)} QA overlays to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    model = Stage1Model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    val_ds = Stage1Dataset(config, "val", augment=False)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False,
                             num_workers=config["train"]["num_workers"])

    metrics = collect_metrics(model, val_loader, config, device)

    print("\n=== Val Dice ===")
    summarize("lesion", metrics["lesion"]["dices"])
    summarize("vessel_artery", metrics["vessel_artery"]["dices"])
    summarize("vessel_vein", metrics["vessel_vein"]["dices"])

    present = metrics["organ_class_present"]
    per_class = metrics["organ_per_class_dice"]
    macro = per_class[present].mean().item() if present.any() else 0.0
    print(f"organ macro dice (over {int(present.sum())} classes present in val): {macro:.4f}")
    print("organ notable classes:")
    for class_id, name in NOTABLE_ORGAN_CLASSES.items():
        if present[class_id]:
            print(f"  {name} (id={class_id}): {per_class[class_id].item():.4f}")
        else:
            print(f"  {name} (id={class_id}): not present in val")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    write_dice_csv(os.path.join(checkpoint_dir, "eval_lesion_dice.csv"),
                    metrics["lesion"]["ids"], metrics["lesion"]["dices"])
    write_dice_csv(os.path.join(checkpoint_dir, "eval_vessel_artery_dice.csv"),
                    metrics["vessel_artery"]["ids"], metrics["vessel_artery"]["dices"])
    write_dice_csv(os.path.join(checkpoint_dir, "eval_vessel_vein_dice.csv"),
                    metrics["vessel_vein"]["ids"], metrics["vessel_vein"]["dices"])

    root_dir = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    qa_dir = config["eval"]["qa_overlay_dir"] if os.path.isabs(config["eval"]["qa_overlay_dir"]) \
        else os.path.join(root_dir, config["eval"]["qa_overlay_dir"])
    save_qa_overlays(model, val_ds, config, device, qa_dir, config["eval"]["num_qa_overlays"])


if __name__ == "__main__":
    main()
