"""
Stage 2 Dataset：讀 data/X-ray/PadChest-GR(-small)/processed/manifest.jsonl，輸出 image + 26 維
multi-label 分類向量。跟 Stage 1 共用同一份官方 split（不重新切 -- Stage 1 的 val/test 對 encoder
來說要維持是「沒看過的資料」，見 STAGE2_PLAN.md 第 1 節）。

image 的載入/normalize/cache 邏輯跟 Stage 1 的 dataset.py 完全一致（同樣的 16-bit PNG per-image
百分位 clip+stretch，同樣把 decode+normalize+resize 的結果 cache 到磁碟 -- 這是 Stage 1 訓練時
實測出來的最大瓶頸，兩個 stage 用同一個 cache 目錄，Stage 1 已經跑過的圖片這裡可以直接吃）。

每筆樣本輸出：
  image: (1, R, R) float32，[-1, 1]
  label: (26,) float32 0/1，順序固定為 config["labels"]
  item_id: study_id
  report_text: 原始報告全文，給 eval.py 做 false negative 人工核對用
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile_normalize(arr, lo_pct, hi_pct):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)  # [0, 1]
    return (disp * 2 - 1).astype(np.float32)  # [-1, 1]


class Stage2Dataset(Dataset):
    def __init__(self, config, split, augment):
        """
        config: 讀進來的 config.yaml dict
        split: "train" / "validation" / "test"（跟 manifest.jsonl 的 `split` 欄位值一致）
        augment: 是否套用 augmentation（train=True，其他=False）
        """
        self.config = config
        self.augment = augment
        self.resolution = config["image"]["resolution"]
        self.pct_lo = config["image"]["percentile_low"]
        self.pct_hi = config["image"]["percentile_high"]
        self.label_names = config["labels"]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        self.data_dir = os.path.join(root_dir, config["data"]["processed_dir"])

        # 跟 Stage 1 用同一套 cache 命名規則（resolution/percentile 一致就是同一份 cache），
        # 兩個 stage 可以互相重用已經算好的圖
        self.cache_dir = os.path.join(
            self.data_dir, f"_cache_r{self.resolution}_p{self.pct_lo}_{self.pct_hi}"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.records = load_jsonl(os.path.join(self.data_dir, "manifest.jsonl"))
        self.records = [r for r in self.records if r["split"] == split]
        if not self.records:
            raise ValueError(f"no rows with split={split!r} in {self.data_dir}/manifest.jsonl")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = self._load_image(r)
        label = np.array(
            [r["classification_labels"][name] for name in self.label_names], dtype=np.float32
        )

        params = self._sample_augment_params() if self.augment else None
        image = self._augment(image[None], params=params)[0]

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label),
            "item_id": r["study_id"],
            "report_text": r.get("report_text", ""),
        }

    def _load_image(self, row):
        cache_path = os.path.join(self.cache_dir, f"{row['study_id']}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        img_path = os.path.join(self.data_dir, row["image_relpath"])
        arr = np.array(Image.open(img_path))
        arr = percentile_normalize(arr, self.pct_lo, self.pct_hi)
        tensor = torch.from_numpy(arr)[None, None]  # (1,1,H,W)
        tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        resized = tensor[0, 0].numpy()

        tmp_path = cache_path + f".tmp{os.getpid()}"
        with open(tmp_path, "wb") as f:
            np.save(f, resized)
        os.replace(tmp_path, cache_path)
        return resized

    def _sample_augment_params(self):
        aug_cfg = self.config["augmentation"]
        return {
            "hflip": random.random() < aug_cfg["hflip_prob"],
            "angle": random.uniform(-aug_cfg["rotation_deg"], aug_cfg["rotation_deg"]),
            "scale": 1.0 + random.uniform(-aug_cfg["scale_jitter"], aug_cfg["scale_jitter"]),
            "intensity_shift": random.uniform(-aug_cfg["intensity_jitter"], aug_cfg["intensity_jitter"]),
        }

    def _augment(self, array, params):
        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, R, R)
        if params is None:
            return tensor[0].numpy()

        if params["hflip"]:
            tensor = torch.flip(tensor, dims=[-1])

        theta = self._affine_theta(params["angle"], params["scale"])
        grid = F.affine_grid(theta, tensor.shape, align_corners=False)
        tensor = F.grid_sample(tensor, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        tensor = tensor * (1 + params["intensity_shift"])

        return tensor[0].numpy()

    @staticmethod
    def _affine_theta(angle_deg, scale):
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor(
            [[cos_a / scale, -sin_a / scale, 0.0], [sin_a / scale, cos_a / scale, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        return theta
