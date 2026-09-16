"""
跑 val/test set，輸出 26 類各自的 AP + GT box 數量（見 STAGE2_5_PLAN.md 第 6 節：box 數太少的
類別 AP 不可靠，要跟數字一起報告），加上 QA overlay（image + GT box + 預測 box）。

跑法：python3 U-VLM/X-ray/cxr/stage2_5/eval.py \
        --encoder-checkpoint U-VLM/X-ray/cxr/stage2_5/checkpoints/best_encoder.pt \
        --head-checkpoint U-VLM/X-ray/cxr/stage2_5/checkpoints/best_head.pt
"""
import argparse
import csv
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import Stage2_5Dataset
from decode import decode_predictions
from model import Stage2_5Model
from train import build_gt_boxes_by_class, get_device, move_batch_to_device, resolve_path, run_validation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_per_label_csv(path, per_label):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n_gt", "ap"])
        for name, m in per_label.items():
            writer.writerow([name, m["n_gt"], "" if m["ap"] is None else round(m["ap"], 4)])


@torch.no_grad()
def save_qa_overlays(model, ds, device, out_dir, num_overlays, topk, num_preds_shown=5):
    """畫每張圖分數最高的 num_preds_shown 個預測，而不是用固定的絕對分數門檻 -- 模型現階段
    （尤其才剛開始訓練、或資料量小）heatmap sigmoid 輸出普遍偏低，固定門檻很容易把所有預測都濾掉，
    畫出一張看起來「模型完全沒預測」的空白圖，但其實只是分數校準的問題，不代表真的沒學到東西。"""
    os.makedirs(out_dir, exist_ok=True)
    from dataset import collect_boxes_and_classes

    for idx in range(min(num_overlays, len(ds))):
        row = ds.records[idx]
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        outputs = model(image)
        dets = decode_predictions(outputs["heatmap"], outputs["size"], outputs["offset"], topk)[0]
        dets.sort(key=lambda d: -d[0])
        dets = dets[:num_preds_shown]

        img_np = ((sample["image"][0].numpy() + 1) / 2 * 255).astype(np.uint8)
        R = img_np.shape[0]
        gt_panel = Image.fromarray(img_np).convert("RGB")
        pred_panel = Image.fromarray(img_np).convert("RGB")

        gt_boxes, _ = collect_boxes_and_classes(row, ds.label_to_group, ds.label_index)
        draw_gt = ImageDraw.Draw(gt_panel)
        for x1, y1, x2, y2 in gt_boxes:
            draw_gt.rectangle([x1 * R, y1 * R, x2 * R, y2 * R], outline=(0, 255, 0), width=2)

        draw_pred = ImageDraw.Draw(pred_panel)
        for score, cls, x1, y1, x2, y2 in dets:
            draw_pred.rectangle([x1 * R, y1 * R, x2 * R, y2 * R], outline=(255, 0, 0), width=2)
            draw_pred.text((x1 * R, max(0, y1 * R - 10)), f"{ds.label_names[cls][:12]} {score:.2f}", fill=(255, 0, 0))

        combined = Image.new("RGB", (R * 3, R))
        combined.paste(Image.fromarray(img_np).convert("RGB"), (0, 0))
        combined.paste(gt_panel, (R, 0))
        combined.paste(pred_panel, (R * 2, 0))
        combined.save(os.path.join(out_dir, f"{row['study_id']}_qa.png"))

    print(f"saved {min(num_overlays, len(ds))} QA overlays to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--encoder-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_encoder.pt"))
    parser.add_argument("--head-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_head.pt"))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    model = Stage2_5Model(config).to(device)
    model.encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
    model.head.load_state_dict(torch.load(args.head_checkpoint, map_location=device))
    model.eval()

    label_names = config["labels"]
    ds = Stage2_5Dataset(config, args.split, augment=False)
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False,
                         num_workers=config["train"]["num_workers"])
    gt_boxes_by_class = build_gt_boxes_by_class(ds, label_names)

    eval_cfg = config["eval"]
    metrics = run_validation(
        model, loader, gt_boxes_by_class, label_names, device,
        eval_cfg["iou_threshold"], eval_cfg["topk"],
    )

    print(f"\n=== {args.split} AP @ IoU>={eval_cfg['iou_threshold']}（每類都附上 GT box 數量）===")
    for name, m in metrics["per_label"].items():
        if m["ap"] is None:
            print(f"  {name}: n_gt={m['n_gt']} -> 這個 split 沒有 GT box，AP 無法計算")
        else:
            print(f"  {name}: n_gt={m['n_gt']} ap={m['ap']:.4f}")
    print(f"\nmacro mAP: {metrics['macro_map']:.4f}")

    checkpoint_dir = os.path.dirname(args.encoder_checkpoint)
    write_per_label_csv(os.path.join(checkpoint_dir, f"eval_per_label_{args.split}.csv"), metrics["per_label"])
    print(f"每類指標存到 {checkpoint_dir}/eval_per_label_{args.split}.csv")

    qa_dir = resolve_path(
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..")), eval_cfg["qa_overlay_dir"]
    )
    save_qa_overlays(model, ds, device, qa_dir, eval_cfg["num_qa_overlays"], eval_cfg["topk"])


if __name__ == "__main__":
    main()
