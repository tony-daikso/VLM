"""
Stage 2 Dataset：讀 processed_rex（CT-RATE）manifest.jsonl，輸出 image + 18 維 multi-label
分類向量。跟 Stage 1 共用同一份 splits.json 的 ctrate train/val 切分（不重新切，理由見
STAGE2_PLAN.md 第 3 節：Stage 1 的 val 對 encoder 來說要維持是「沒看過的資料」）。

HiPaS 沒有 classification_labels（見 STAGE2_PLAN.md 第 1 節），Stage 2 完全用不到，
所以這裡跟 Stage 1 的 dataset.py 不同，只讀 processed_rex 一份 manifest。

每筆樣本輸出：
  image: (1, R, R) float32，肺窗 windowing 後 normalize 到 [-1, 1]（跟 Stage 1 完全一致）
  label: (num_labels,) float32 0/1，順序固定為 config["labels"]
  item_id: volume_id
  report_text: 原始報告全文，給 eval.py 做 false negative 人工核對用
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_report_text(report_text):
    """manifest 裡 report_text 是 dict（ClinicalInformation_EN/Technique_EN/Findings_EN/
    Impressions_EN），只取 Findings/Impressions 兩段接起來給 eval.py 的 false negative 人工核對用，
    Clinical/Technique 是檢查方式的制式描述，跟病理判斷無關。"""
    if not isinstance(report_text, dict):
        return str(report_text or "")
    parts = [report_text.get("Findings_EN", ""), report_text.get("Impressions_EN", "")]
    return " | ".join(p for p in parts if p)


def window_normalize(image_hu, window_level, window_width):
    """HU 值 -> 肺窗 windowing -> normalize 到 [-1, 1]。"""
    lo = window_level - window_width / 2
    hi = window_level + window_width / 2
    image = np.clip(image_hu, lo, hi)
    image = (image - lo) / (hi - lo)  # [0, 1]
    image = image * 2 - 1  # [-1, 1]
    return image.astype(np.float32)


class Stage2Dataset(Dataset):
    def __init__(self, config, split, augment):
        """
        config: 讀進來的 config.yaml dict
        split: "train" 或 "val"
        augment: 是否套用 augmentation（train=True, val=False）
        """
        self.config = config
        self.split = split
        self.augment = augment
        self.resolution = config["image"]["resolution"]
        self.window_level = config["image"]["window_level"]
        self.window_width = config["image"]["window_width"]
        self.label_names = config["labels"]

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rex_dir = os.path.join(root_dir, config["data"]["processed_rex_dir"])

        splits_path = os.path.join(root_dir, config["data"]["splits_path"])
        with open(splits_path) as f:
            splits = json.load(f)
        ctrate_ids = set(splits["ctrate"][split])

        records = load_jsonl(os.path.join(self.rex_dir, "manifest.jsonl"))
        self.records = sorted(
            (r for r in records if r["volume_id"] in ctrate_ids), key=lambda r: r["volume_id"]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = np.load(os.path.join(self.rex_dir, r["image_npy"]))
        image = window_normalize(image, self.window_level, self.window_width)

        label = np.array(
            [r["classification_labels"][name] for name in self.label_names], dtype=np.float32
        )

        params = self._sample_augment_params() if self.augment else None
        image = self._resize(image[None], mode="bilinear", params=params)[0]

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label),
            "item_id": r["volume_id"],
            "report_text": format_report_text(r.get("report_text")),
        }

    def _sample_augment_params(self):
        aug_cfg = self.config["augmentation"]
        return {
            "hflip": random.random() < aug_cfg["hflip_prob"],
            "angle": random.uniform(-aug_cfg["rotation_deg"], aug_cfg["rotation_deg"]),
            "scale": 1.0 + random.uniform(-aug_cfg["scale_jitter"], aug_cfg["scale_jitter"]),
            "intensity_shift": random.uniform(-aug_cfg["intensity_jitter"], aug_cfg["intensity_jitter"]),
        }

    def _resize(self, array, mode, params):
        """array: (C, H, W) numpy -> (C, R, R) numpy，做完 resize 後如果有 augment 參數就套用
        flip/rotate/scale/intensity jitter（跟 Stage 1 dataset.py 的 _resize 邏輯一致）。"""
        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, H, W)
        tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode=mode, align_corners=False)

        if params is not None:
            if params["hflip"]:
                tensor = torch.flip(tensor, dims=[-1])

            theta = self._affine_theta(params["angle"], params["scale"])
            grid = F.affine_grid(theta, tensor.shape, align_corners=False)
            tensor = F.grid_sample(tensor, grid, mode=mode, align_corners=False, padding_mode="zeros")
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
