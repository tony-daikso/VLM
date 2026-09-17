"""
用 Phase 2 訓練好的 DINO encoder，把 PadChest-GR 每個 study 的胸腔 X-ray 轉成學長那套「馬賽克
假圖」，再存成 PNG，餵給 Phase 4 的 LLaVA 訓練/Phase 5 的推論。

演算法完全照抄 DINO_LLM/prior/inference.py 的 dino_inference()/to_uint8_zscore()：
CLS token(1280 維) reshape 成 5 個 16x16 block，貼進 final_size x final_size 的畫布，
z-score 轉 uint8。

跟原本 CT 版本的差異（必要調整，非隨意更動）：原版是給「一個 study 資料夾裡一堆切面 PNG」用的，
每張切面貢獻 5 個 block，可以把整張畫布塞滿；X-ray 一個 study 只有一張 2D 影像，所以每張假圖
只會填進畫布左上角 5/900 格，其餘維持 0（等於 z-score 後的中性灰）。這點會在最終比較報告中
明確記錄。

用法：
    python3 build_mosaics.py \
        --manifest /root/Desktop/VLM/data/X-ray/PadChest-GR/processed/manifest.jsonl \
        --data_root /root/Desktop/VLM/data/X-ray/PadChest-GR/processed \
        --dino_checkpoint /datadrive/VLM/DINO_LLM/X-ray/stage_dino_ssl/checkpoints/checkpoint.pth \
        --output_dir /datadrive/VLM/DINO_LLM/X-ray/mosaics
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dino.dino_model_c import load_dino_model

FINAL_SIZE = 16 * 30  # 480，跟學長 llava_webhook.py 的 final_size 一致
IMG_SIZE = 512


def to_uint8_zscore(x, k=3, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    z = (x - mu) / (sd + eps)
    z = np.clip(z, -k, k)
    y = (z + k) / (2 * k)
    return (y * 255.0 + 0.5).astype(np.uint8)


def build_one_mosaic(model, img_path, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image = transform(image)
    with torch.no_grad():
        emb = model(image.unsqueeze(0).to(device)).cpu().numpy().reshape(1, -1).astype("float16")
    emb = emb.reshape(5, 16, 16)

    merge_img = np.zeros([FINAL_SIZE, FINAL_SIZE])
    for m in range(5):
        row, col = divmod(m, FINAL_SIZE // 16)
        merge_img[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = emb[m]

    dino_img = to_uint8_zscore(merge_img)
    dino_img[np.where(merge_img == 0)] = 0
    return Image.fromarray(dino_img).convert("RGB")


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dino_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino_model(args.dino_checkpoint)

    rows = load_manifest(args.manifest)
    for split in args.splits:
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        split_rows = [r for r in rows if r["split"] == split]
        for i, row in enumerate(split_rows):
            out_path = os.path.join(out_dir, f"{row['study_id']}.png")
            if os.path.exists(out_path):
                continue
            img_path = os.path.join(args.data_root, row["image_relpath"])
            mosaic = build_one_mosaic(model, img_path, device)
            mosaic.save(out_path)
            if (i + 1) % 200 == 0:
                print(f"[{split}] {i + 1}/{len(split_rows)}")
        print(f"[{split}] done, {len(split_rows)} mosaics -> {out_dir}")


if __name__ == "__main__":
    main()
