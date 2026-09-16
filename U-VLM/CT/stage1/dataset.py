"""
Stage 1 Dataset：讀 processed_rex（CT-RATE + ReXGroundingCT + TotalSegmentator）跟
processed_hipas（HiPaS 血管）兩份 manifest.jsonl，統一成帶 `source` 標記的樣本。

每筆樣本輸出：
  image: (1, R, R) float32，肺窗 windowing 後 normalize 到 [-1, 1]
  source: "ctrate" 或 "hipas"
  lesion_mask: (1, R, R) float32 binary（ctrate 樣本才有意義，hipas 樣本填 0 且 valid_lesion_organ=False）
  organ_mask: (R, R) int64，值域 0~117（ctrate 樣本才有意義）
  vessel_mask: (2, R, R) float32 binary，artery/vein（hipas 樣本才有意義，ctrate 樣本填 0 且 valid_vessel=False）
  valid_lesion_organ: bool，這筆樣本的 lesion/organ loss 要不要算
  valid_vessel: bool，這筆樣本的 vessel loss 要不要算

Train/val 用同一組隨機參數對 image 跟所有 mask 做 augmentation（見 _augment）。
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NUM_ORGAN_CLASSES = 118  # TotalSegmentator id 0(背景)~117


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def window_normalize(image_hu, window_level, window_width):
    """HU 值 -> 肺窗 windowing -> normalize 到 [-1, 1]。"""
    lo = window_level - window_width / 2
    hi = window_level + window_width / 2
    image = np.clip(image_hu, lo, hi)
    image = (image - lo) / (hi - lo)  # [0, 1]
    image = image * 2 - 1  # [-1, 1]
    return image.astype(np.float32)


class Stage1Dataset(Dataset):
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

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rex_dir = os.path.join(root_dir, config["data"]["processed_rex_dir"])
        self.hipas_dir = os.path.join(root_dir, config["data"]["processed_hipas_dir"])

        splits_path = os.path.join(root_dir, config["data"]["splits_path"])
        with open(splits_path) as f:
            splits = json.load(f)
        ctrate_ids = set(splits["ctrate"][split])
        hipas_ids = set(splits["hipas"][split])

        rex_records = load_jsonl(os.path.join(self.rex_dir, "manifest.jsonl"))
        self.rex_by_id = {r["volume_id"]: r for r in rex_records if r["volume_id"] in ctrate_ids}

        hipas_records = load_jsonl(os.path.join(self.hipas_dir, "manifest.jsonl"))
        self.hipas_by_id = {r["case_id"]: r for r in hipas_records if r["case_id"] in hipas_ids}

        self.items = (
            [("ctrate", vid) for vid in sorted(self.rex_by_id)]
            + [("hipas", cid) for cid in sorted(self.hipas_by_id)]
        )

    def __len__(self):
        return len(self.items)

    def source_flags(self):
        """給 train.py 的 WeightedRandomSampler 用：回傳跟 self.items 對齊的 source list。"""
        return [source for source, _ in self.items]

    def __getitem__(self, idx):
        source, item_id = self.items[idx]
        if source == "ctrate":
            sample = self._load_ctrate(item_id)
        else:
            sample = self._load_hipas(item_id)

        params = self._sample_augment_params() if self.augment else None
        sample["image"] = self._resize(sample["image"][None], mode="bilinear", params=params)[0]
        sample["lesion_mask"] = self._resize(sample["lesion_mask"], mode="nearest", params=params)
        sample["organ_mask"] = self._resize(
            sample["organ_mask"][None].astype(np.float32), mode="nearest", params=params
        )[0].astype(np.int64)
        sample["vessel_mask"] = self._resize(sample["vessel_mask"], mode="nearest", params=params)

        return {
            "image": torch.from_numpy(sample["image"]).unsqueeze(0),
            "source": source,
            "lesion_mask": torch.from_numpy(sample["lesion_mask"]),
            "organ_mask": torch.from_numpy(sample["organ_mask"]),
            "vessel_mask": torch.from_numpy(sample["vessel_mask"]),
            "valid_lesion_organ": source == "ctrate",
            "valid_vessel": source == "hipas",
            "item_id": item_id,
        }

    def _load_ctrate(self, volume_id):
        r = self.rex_by_id[volume_id]
        image = np.load(os.path.join(self.rex_dir, r["image_npy"]))
        image = window_normalize(image, self.window_level, self.window_width)

        mask_npz = np.load(os.path.join(self.rex_dir, r["segmentation_mask_npz"]))
        lesion = mask_npz["mask"]  # (N, H, W), 值可能 >1（重疊 instance 疊加）
        lesion_binary = (lesion.sum(axis=0) > 0).astype(np.float32)[None]  # (1, H, W)

        organ = np.load(os.path.join(self.rex_dir, "organ_masks", f"{volume_id}.npy")).astype(np.int64)

        h, w = image.shape
        vessel = np.zeros((2, h, w), dtype=np.float32)

        return {
            "image": image,
            "lesion_mask": lesion_binary,
            "organ_mask": organ,
            "vessel_mask": vessel,
        }

    def _load_hipas(self, case_id):
        r = self.hipas_by_id[case_id]
        image = np.load(os.path.join(self.hipas_dir, r["image_npy"]))
        image = window_normalize(image, self.window_level, self.window_width)

        mask_npz = np.load(os.path.join(self.hipas_dir, r["vessel_mask_npz"]))
        vessel = np.stack([mask_npz["artery"], mask_npz["vein"]], axis=0).astype(np.float32)  # (2, H, W)

        h, w = image.shape
        lesion = np.zeros((1, h, w), dtype=np.float32)
        organ = np.zeros((h, w), dtype=np.int64)

        return {
            "image": image,
            "lesion_mask": lesion,
            "organ_mask": organ,
            "vessel_mask": vessel,
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
        flip/rotate/scale（image 跟所有 mask 用同一組參數，維持像素對應關係）。"""
        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, H, W)
        tensor = F.interpolate(
            tensor, size=(self.resolution, self.resolution), mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )

        if params is not None:
            if params["hflip"]:
                tensor = torch.flip(tensor, dims=[-1])

            theta = self._affine_theta(params["angle"], params["scale"])
            grid = F.affine_grid(theta, tensor.shape, align_corners=False)
            tensor = F.grid_sample(tensor, grid, mode=mode, align_corners=False, padding_mode="zeros")

            if mode == "bilinear":
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
