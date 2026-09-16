"""
Stage 1 Dataset：讀 data/X-ray/PadChest-GR(-small)/processed/manifest.jsonl（單一來源，
不像 CT 版要合併 processed_rex + processed_hipas 兩份 manifest -- PadChest-GR 只有一個
lesion head）。

每筆樣本輸出：
  image: (1, R, R) float32，16-bit 原圖 per-image 百分位 clip+stretch 後 normalize 到 [-1, 1]
  lesion_mask: (1, R, R) float32 binary，把這個 study 所有 finding 的 box（0~1 normalized
    座標）直接在 R×R 網格上 rasterize 成矩形前景後 union 起來 -- 這是弱監督（box 不是病灶
    輪廓），normal / 沒有可定位 box 的 study 就是全 0 mask，一樣參與訓練提供負樣本監督
    （見 STAGE1_PLAN.md 第 2 節）

split 直接用 manifest.jsonl 裡的官方 `split` 欄位（"train"/"validation"/"test"），不像 CT
版需要額外的 splits.json -- PadChest-GR 本身就有 patient-safe 的官方切分。

Train 用同一組隨機參數對 image 跟 mask 做 augmentation（見 _augment），跟 CT 版手法一致。
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
    """16-bit 原圖 -> per-image 百分位 clip + min-max stretch -> [-1, 1]（見
    scripts/make_qa_overlays_padchest_gr.py 裡驗證過的同一套邏輯，PIL 的 naive
    convert("L") 在 16-bit 影像上會直接 clip 出洗白的圖，不能用）。"""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)  # [0, 1]
    return (disp * 2 - 1).astype(np.float32)  # [-1, 1]


def boxes_to_mask(findings, resolution):
    """box 座標是 0~1 normalized，跟原圖解析度無關，直接在 R×R 網格上畫矩形，
    不需要先在原圖解析度畫好再 resize。"""
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    for finding in findings:
        for box in finding["boxes"]:
            x1, y1, x2, y2 = box
            xi1, xi2 = sorted((x1, x2))
            yi1, yi2 = sorted((y1, y2))
            xi1 = max(0, min(resolution, int(round(xi1 * resolution))))
            xi2 = max(0, min(resolution, int(round(xi2 * resolution))))
            yi1 = max(0, min(resolution, int(round(yi1 * resolution))))
            yi2 = max(0, min(resolution, int(round(yi2 * resolution))))
            mask[yi1:yi2, xi1:xi2] = 1.0
    return mask[None]  # (1, R, R)


class Stage1Dataset(Dataset):
    def __init__(self, config, split, augment, limit=None):
        """
        config: 讀進來的 config.yaml dict
        split: "train" / "validation" / "test"（跟 manifest.jsonl 的 `split` 欄位值一致）
        augment: 是否套用 augmentation（train=True，其他=False）
        limit: 只取前 N 筆（smoke test 用，不影響完整訓練，預設 None = 不限制）
        """
        self.config = config
        self.augment = augment
        self.resolution = config["image"]["resolution"]
        self.pct_lo = config["image"]["percentile_low"]
        self.pct_hi = config["image"]["percentile_high"]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        self.data_dir = os.path.join(root_dir, config["data"]["processed_dir"])

        records = load_jsonl(os.path.join(self.data_dir, "manifest.jsonl"))
        self.rows = [r for r in records if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows with split={split!r} in {self.data_dir}/manifest.jsonl")
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self._load_image(row)
        lesion_mask = boxes_to_mask(row["findings"], self.resolution)

        params = self._sample_augment_params() if self.augment else None
        image = self._augment(image[None], mode="bilinear", params=params, is_image=True)[0]
        lesion_mask = self._augment(lesion_mask, mode="nearest", params=params, is_image=False)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "lesion_mask": torch.from_numpy(lesion_mask),
            "study_id": row["study_id"],
        }

    def _load_image(self, row):
        img_path = os.path.join(self.data_dir, row["image_relpath"])
        arr = np.array(Image.open(img_path))
        arr = percentile_normalize(arr, self.pct_lo, self.pct_hi)
        tensor = torch.from_numpy(arr)[None, None]  # (1,1,H,W)
        tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return tensor[0, 0].numpy()

    def _sample_augment_params(self):
        aug_cfg = self.config["augmentation"]
        return {
            "hflip": random.random() < aug_cfg["hflip_prob"],
            "angle": random.uniform(-aug_cfg["rotation_deg"], aug_cfg["rotation_deg"]),
            "scale": 1.0 + random.uniform(-aug_cfg["scale_jitter"], aug_cfg["scale_jitter"]),
            "intensity_shift": random.uniform(-aug_cfg["intensity_jitter"], aug_cfg["intensity_jitter"]),
        }

    def _augment(self, array, mode, params, is_image):
        """array: (C, R, R) numpy，已經是目標解析度 -> 套用 flip/rotate/scale
        （image 跟 mask 用同一組參數，維持像素對應關係），image 額外做 intensity jitter。"""
        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, R, R)
        if params is None:
            return tensor[0].numpy()

        if params["hflip"]:
            tensor = torch.flip(tensor, dims=[-1])

        theta = self._affine_theta(params["angle"], params["scale"])
        grid = F.affine_grid(theta, tensor.shape, align_corners=False)
        tensor = F.grid_sample(tensor, grid, mode=mode, align_corners=False, padding_mode="zeros")

        if is_image:
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
