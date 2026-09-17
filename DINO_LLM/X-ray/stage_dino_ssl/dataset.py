"""
PadChest-GR train split 的 DINO multi-crop 資料集。只用 train split（3,185 張，unlabeled，
self-supervised 不需要 report/label），manifest.jsonl 路徑跟欄位跟 U-VLM 的
X-ray/cxr/stage3/dataset.py 完全一樣（study_id / image_relpath / split）。

跟原版 facebookresearch/dino 的 DataAugmentationDINO 相比，兩個刻意的調整：
1. 影像先用 U-VLM 同款 percentile normalize（0.5~99.5 分位數 clip + min-max stretch）轉成
   uint8 灰階，再複製成 3 channel 餵給 ImageNet 預設的 ViT（in_chans=3），而不是直接假設
   自然圖片的 0-255 RGB 讀法。
2. 拿掉 RandomHorizontalFlip：胸腔 X-ray 的左右不對稱（心臟位置、主動脈弓等），水平翻轉會
   產生不存在的解剖結構，對醫學影像的自監督學習是無效的 augmentation。
"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def load_manifest(manifest_path, split):
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["split"] == split:
                records.append(row)
    return records


def percentile_normalize_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 255.0).astype(np.uint8)


class GaussianBlur(object):
    """跟官方 dino/utils.py 裡的 GaussianBlur 一致，這裡獨立放一份方便 dataset.py 自己 import。"""

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


class Solarization(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class DataAugmentationDINO(object):
    """比照官方 main_dino.py 的 DataAugmentationDINO，拿掉 RandomHorizontalFlip 跟
    ColorJitter/Grayscale（輸入本來就是灰階醫學影像，不需要顏色抖動），其餘（多重 crop 尺度、
    global/local crop 數量、GaussianBlur/Solarization 機率）沿用官方預設值。"""

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 global_crop_size=224, local_crop_size=96):
        flip_and_jitter = transforms.Compose([])  # 保留這個位置只是為了跟官方結構對照，內容刻意留空

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


CACHE_RESOLUTION = 320  # 比 global crop(224)大、比原圖(~1824x1652)小很多，讓 RandomResizedCrop
                        # 還留有 zoom 的空間，同時把「decode 大圖 + 算 percentile」的成本從每個
                        # epoch每次都做，降到只需要做一次（見下方 cache 邏輯）。


class PadChestSSLDataset(Dataset):
    """
    第一次跑 smoke test 時發現：每個 sample 的 __getitem__（PIL decode 原圖 1824x1652 +
    percentile normalize + 8 次 RandomResizedCrop/GaussianBlur）平均要 ~0.48 秒，导致 GPU
    大部分時間在等 dataloader（nvidia-smi 常看到 0% 但其實訓練沒有卡住，只是 CPU-bound）。
    這裡加一層磁碟 cache：把每張圖 percentile-normalize 完、縮到 CACHE_RESOLUTION 大小後存成
    PNG，之後同一張圖只需要付一次這個成本，跟 U-VLM dataset.py 的 cache 慣例一致。
    """

    def __init__(self, manifest_path, data_root, transform):
        self.records = load_manifest(manifest_path, split="train")
        self.data_root = data_root
        self.transform = transform
        self.cache_dir = os.path.join(data_root, f"_cache_dino_ssl_r{CACHE_RESOLUTION}")
        os.makedirs(self.cache_dir, exist_ok=True)
        if not self.records:
            raise ValueError(f"no train-split rows found in {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        cache_path = os.path.join(self.cache_dir, f"{row['study_id']}.png")
        if os.path.exists(cache_path):
            image = Image.open(cache_path).convert("RGB")
        else:
            img_path = os.path.join(self.data_root, row["image_relpath"])
            arr = np.array(Image.open(img_path))
            arr = percentile_normalize_uint8(arr)
            image = Image.fromarray(arr, mode="L").resize(
                (CACHE_RESOLUTION, CACHE_RESOLUTION), Image.BICUBIC
            )
            tmp_path = cache_path + f".tmp{os.getpid()}"
            image.save(tmp_path, format="PNG")
            os.replace(tmp_path, cache_path)
            image = image.convert("RGB")
        crops = self.transform(image)
        return crops
