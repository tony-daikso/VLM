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

另一個必要調整：PadChest-GR 原圖是 16-bit PNG（PIL mode "I;16"），跟學長原本 CT 版本已經先做過
HU windowing + 轉 8-bit 的來源不一樣。一開始漏掉這步，直接 `Image.open().convert("RGB")` 會把
16-bit 像素值naive 轉換、幾乎全部夾到接近全白（實測平均值 241.95/255），等於 DINO 看到的是張
壞掉的圖。這裡改成跟 `stage_dino_ssl/dataset.py` 一致的 percentile normalize 流程。

第三個調整（效能）：一開始是單執行緒一張張處理，decode 大圖 + percentile normalize 這步是
CPU-bound，跑起來很慢卻沒善用這台機器的 12 核心。改成跟 Phase 2 訓練一樣，用 DataLoader +
多個 worker 平行做 CPU 前處理，GPU 端則批次(batch)跑 DINO forward，兩邊重疊執行。

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dino.dino_model_c import load_dino_model
from stage_dino_ssl.dataset import percentile_normalize_uint8

FINAL_SIZE = 16 * 30  # 480，跟學長 llava_webhook.py 的 final_size 一致
IMG_SIZE = 512
IMG_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def to_uint8_zscore(x, k=3, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    z = (x - mu) / (sd + eps)
    z = np.clip(z, -k, k)
    y = (z + k) / (2 * k)
    return (y * 255.0 + 0.5).astype(np.uint8)


def embedding_to_mosaic(emb):
    """emb: (1280,) float -> 480x480 RGB PIL Image，跟 dino_inference() 的邏輯一致。"""
    emb = emb.reshape(5, 16, 16)
    merge_img = np.zeros([FINAL_SIZE, FINAL_SIZE])
    for m in range(5):
        row, col = divmod(m, FINAL_SIZE // 16)
        merge_img[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = emb[m]
    dino_img = to_uint8_zscore(merge_img)
    dino_img[np.where(merge_img == 0)] = 0
    return Image.fromarray(dino_img).convert("RGB")


class MosaicSourceDataset(Dataset):
    """負責 CPU-bound 的那段（decode 16-bit 原圖 + percentile normalize + resize），
    交給 DataLoader 的多個 worker 平行做，GPU 端的 forward 在主行程批次跑。"""

    def __init__(self, rows, data_root):
        self.rows = rows
        self.data_root = data_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.data_root, row["image_relpath"])
        arr = percentile_normalize_uint8(np.array(Image.open(img_path)))
        image = Image.fromarray(arr, mode="L").convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return row["study_id"], IMG_TRANSFORM(image)


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
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino_model(args.dino_checkpoint)

    rows = load_manifest(args.manifest)
    for split in args.splits:
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        split_rows = [r for r in rows if r["split"] == split]
        pending = [r for r in split_rows if not os.path.exists(os.path.join(out_dir, f"{r['study_id']}.png"))]
        if not pending:
            print(f"[{split}] 全部 {len(split_rows)} 筆都已經有假圖，跳過")
            continue

        loader = DataLoader(
            MosaicSourceDataset(pending, args.data_root),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        )
        done = 0
        with torch.no_grad():
            for study_ids, images in loader:
                embs = model(images.to(device)).cpu().numpy().astype("float16")
                for study_id, emb in zip(study_ids, embs):
                    mosaic = embedding_to_mosaic(emb)
                    mosaic.save(os.path.join(out_dir, f"{study_id}.png"))
                done += len(study_ids)
                if done % 200 < args.batch_size:
                    print(f"[{split}] {done}/{len(pending)}")
        print(f"[{split}] done, {len(split_rows)} mosaics -> {out_dir}")


if __name__ == "__main__":
    main()
