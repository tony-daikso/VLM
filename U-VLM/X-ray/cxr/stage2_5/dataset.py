"""
Stage 2.5 Dataset：讀 data/X-ray/PadChest-GR(-small)/processed/manifest.jsonl +
label_to_label_group.json，輸出 image + CenterNet 風格的 heatmap/size/offset/reg_mask target
（見 STAGE2_5_PLAN.md 第 1、2、3 節）。

image 的載入/normalize/cache 邏輯跟 Stage 1/2 完全一致（同一個 cache 目錄，三個 stage 互相
重用已經算好的圖）。

Box 的 augmentation 處理方式（見 STAGE2_5_PLAN.md 第 3 節）：每個 box 各自 rasterize 成一張獨立
mask，跟 image 一起過同一個 affine_grid/grid_sample，再從 warp 後的 mask 反推增強後的 box 座標
（旋轉/縮放到完全出界就丟棄這個 box）--这样不用手動推導矩形四個角座標的解析變換。

每筆樣本輸出：
  image: (1, R, R) float32，[-1, 1]
  heatmap: (26, F, F) float32
  size_target: (2, F, F) float32
  offset_target: (2, F, F) float32
  reg_mask: (1, F, F) float32
  study_id: str
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from target import encode_targets


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
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 2 - 1).astype(np.float32)


def collect_boxes_and_classes(row, label_to_group, label_index):
    """回傳 (boxes, class_index_lists)，boxes 是原始（未增強）0~1 normalized (x1,y1,x2,y2) 列表，
    class_index_lists 跟 boxes 等長，每個元素是這個 box 對應的 label_group index 列表（可能 >1 個，
    見 STAGE2_5_PLAN.md 第 1 節「同一個 finding 底下多個 fine-grained label 映射到多個 group」）。"""
    boxes, class_index_lists = [], []
    for finding in row["findings"]:
        if not finding["boxes"]:
            continue
        groups = sorted({label_index[label_to_group[l]] for l in finding["labels"] if l in label_to_group})
        if not groups:
            continue
        for box in finding["boxes"]:
            boxes.append(tuple(box))
            class_index_lists.append(groups)
    return boxes, class_index_lists


class Stage2_5Dataset(Dataset):
    def __init__(self, config, split, augment):
        self.config = config
        self.augment = augment
        self.resolution = config["image"]["resolution"]
        self.pct_lo = config["image"]["percentile_low"]
        self.pct_hi = config["image"]["percentile_high"]
        self.feature_size = self.resolution // config["feature_stride"]
        self.label_names = config["labels"]
        self.label_index = {name: i for i, name in enumerate(self.label_names)}
        self.min_overlap = config["loss"]["gaussian_min_overlap"]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        self.data_dir = os.path.join(root_dir, config["data"]["processed_dir"])

        # 跟 Stage 1/2 用同一套 cache 命名規則，互相重用
        self.cache_dir = os.path.join(
            self.data_dir, f"_cache_r{self.resolution}_p{self.pct_lo}_{self.pct_hi}"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        with open(os.path.join(self.data_dir, "label_to_label_group.json")) as f:
            self.label_to_group = json.load(f)

        self.records = load_jsonl(os.path.join(self.data_dir, "manifest.jsonl"))
        self.records = [r for r in self.records if r["split"] == split]
        if not self.records:
            raise ValueError(f"no rows with split={split!r} in {self.data_dir}/manifest.jsonl")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image = self._load_image(row)
        boxes, class_index_lists = collect_boxes_and_classes(row, self.label_to_group, self.label_index)

        params = self._sample_augment_params() if self.augment else None
        image, boxes, class_index_lists = self._augment(image, boxes, class_index_lists, params)

        heatmap, size_target, offset_target, reg_mask = encode_targets(
            boxes, class_index_lists, self.feature_size, len(self.label_names), self.min_overlap
        )

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "heatmap": torch.from_numpy(heatmap),
            "size_target": torch.from_numpy(size_target),
            "offset_target": torch.from_numpy(offset_target),
            "reg_mask": torch.from_numpy(reg_mask),
            "study_id": row["study_id"],
        }

    def _load_image(self, row):
        cache_path = os.path.join(self.cache_dir, f"{row['study_id']}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        img_path = os.path.join(self.data_dir, row["image_relpath"])
        arr = np.array(Image.open(img_path))
        arr = percentile_normalize(arr, self.pct_lo, self.pct_hi)
        tensor = torch.from_numpy(arr)[None, None]
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

    def _augment(self, image, boxes, class_index_lists, params):
        if params is None or not boxes:
            if params is not None and params["intensity_shift"]:
                image = image * (1 + params["intensity_shift"])
            return image, boxes, class_index_lists

        R = self.resolution
        theta = self._affine_theta(params["angle"], params["scale"])

        image_t = torch.from_numpy(image)[None, None]  # (1,1,R,R)
        if params["hflip"]:
            image_t = torch.flip(image_t, dims=[-1])
        grid = F.affine_grid(theta, image_t.shape, align_corners=False)
        image_t = F.grid_sample(image_t, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        image_t = image_t * (1 + params["intensity_shift"])
        image_out = image_t[0, 0].numpy()

        # 每個 box 各自 rasterize 成一張獨立 mask，跟同一組 grid 一起 warp
        n = len(boxes)
        masks = np.zeros((n, R, R), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            xi1, xi2 = sorted((int(round(x1 * R)), int(round(x2 * R))))
            yi1, yi2 = sorted((int(round(y1 * R)), int(round(y2 * R))))
            xi1, xi2 = max(0, xi1), min(R, xi2)
            yi1, yi2 = max(0, yi1), min(R, yi2)
            masks[i, yi1:yi2, xi1:xi2] = 1.0

        masks_t = torch.from_numpy(masks)[None]  # (1, n, R, R)
        if params["hflip"]:
            masks_t = torch.flip(masks_t, dims=[-1])
        masks_t = F.grid_sample(masks_t, grid, mode="nearest", align_corners=False, padding_mode="zeros")
        masks_out = masks_t[0].numpy()  # (n, R, R)

        new_boxes, new_classes = [], []
        for i in range(n):
            ys, xs = np.where(masks_out[i] > 0.5)
            if len(xs) == 0:
                continue  # 被旋轉/縮放到完全出界，丟棄
            new_boxes.append((xs.min() / R, ys.min() / R, (xs.max() + 1) / R, (ys.max() + 1) / R))
            new_classes.append(class_index_lists[i])

        return image_out, new_boxes, new_classes

    @staticmethod
    def _affine_theta(angle_deg, scale):
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor(
            [[cos_a / scale, -sin_a / scale, 0.0], [sin_a / scale, cos_a / scale, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        return theta
