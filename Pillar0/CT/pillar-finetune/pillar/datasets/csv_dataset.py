import os
import json
import ast
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pillar.datasets.nlst import CENSORING_DIST

import rve


class CSVDataset(torch.utils.data.Dataset):
    """
    Generic CSV-driven dataset for fine-tuning.

    Expected CSV columns:
    - accession: unique identifier per sample (string/int)
    - image_paths: list of RVE paths (JSON list string like ["...", "..."], or a string
      delimited by one of '|', ';', ','). Each path will be loaded via rve.load_sample and
      concatenated along the channel dimension.
    - mask_path: optional RVE path to the segmentation (same spatial dims as image). If present
      and exists on disk, the mask will be loaded; otherwise a zero mask is returned.
    - Any additional columns are included in the returned sample unchanged (no type or value modification).

    Returns a dict per item with at least:
    - x: torch.Tensor with shape (C, D=num_images, H, W) after symmetric padding along D
    - mask: torch.Tensor with shape (1, D=num_images, H, W). Zero if mask_path missing/not found
    - has_annotation: bool
    - image_annotations: torch.Tensor with soft mask (mask / mask.sum()) when mask present, otherwise zeros
    - accession: same as input CSV
    - All additional label columns from the CSV (unchanged)
    - sample_name: mirrored from accession for convenience
    - anatomy: optional passthrough attribute if provided
    """

    def __init__(
        self,
        args,
        augmentations,
        num_images: int = 192,
        max_followup: int = 6,
        csv_path: str = None,
        anatomy: Optional[str] = None,
        split_group="train",
        windows: Optional[str] = "default",
        img_size: Optional[List[int]] = None,
    ) -> None:
        self.csv_path = csv_path
        self.num_images = num_images
        self.max_followup = max_followup
        self.augmentations = augmentations
        self.anatomy = anatomy
        self.img_size = img_size

        df = pd.read_csv(self.csv_path)
        df = df[df["split"] == split_group].reset_index(drop=True)
        required_cols = ["accession", "image_paths"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'")
        # mask_path is optional
        self.has_mask_col = "mask_path" in df.columns

        self.df = df.reset_index(drop=True)
        self.windows = windows

        # Identify columns to passthrough unchanged
        self.label_columns: List[str] = [
            c for c in self.df.columns if c not in ["accession", "image_paths", "mask_path", "split"]
        ]
        self.info = {}
        if split_group == "train":
            # Note that even for evaluation on dev, we want to use the censoring distribution on training set.
            # On test it depends. For now, we are also using the censoring distribution on training set.

            censoring_distribution = CENSORING_DIST

            self.info["censoring_distribution"] = censoring_distribution

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _parse_image_paths(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, (float, int)):
            # Unexpected scalar, treat as single path-string
            return [str(value)]
        if not isinstance(value, str):
            return [str(value)]

        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
            except json.JSONDecodeError:
                pass

        # Fall back to common delimiters
        for delim in ["|", ";", ","]:
            if delim in text:
                parts = [p.strip() for p in text.split(delim) if p.strip()]
                if parts:
                    return parts

        return [text]

    def _pad_along_depth(self, tensor: torch.Tensor, target_depth: int) -> torch.Tensor:
        """
        Pad a (C, D, H, W) or (D, H, W) tensor symmetrically along D to target_depth.
        """
        if tensor.dim() == 3:
            depth = tensor.shape[0]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[crop_left:]
                return tensor[crop_left:-crop_right]
            return tensor
        elif tensor.dim() == 4:
            depth = tensor.shape[1]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[:, crop_left:]
                return tensor[:, crop_left:-crop_right]
            return tensor
        else:
            raise ValueError(f"Unexpected tensor shape for D padding: {tuple(tensor.shape)}")

    def _crop_side(self, tensor: torch.Tensor, target_size: List[int]) -> torch.Tensor:
        """
        Crop a (D, H, W) tensor symmetrically along H and W to target_size.
        """
        H, W = tensor.shape[2:]
        if H > target_size[0]:
            crop_side = (H - target_size[0]) // 2
            tensor = tensor[:, :, crop_side:-crop_side, :]
        if W > target_size[1]:
            crop_side = (W - target_size[1]) // 2
            tensor = tensor[:, :, :, crop_side:-crop_side]
        return tensor

    def _compute_survival_labels(self, y: int, time_at_event: int):
        """
        Compute y_seq and y_mask from y and time_at_event.
        Matches the logic in NLSTDataset.get_label().

        y_seq: binary array of length max_followup, 1s from time_at_event onwards if y=1
        y_mask: binary array of length max_followup, 1s for positions 0 to time_at_event
        """
        y_seq = np.zeros(self.max_followup)
        if y:
            y_seq[time_at_event:] = 1
        y_mask = np.array([1] * (time_at_event + 1) + [0] * (self.max_followup - (time_at_event + 1)))
        return y_seq.astype("float64"), y_mask.astype("float64")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        accession_value = row["accession"]
        accession = str(int(accession_value)) if not isinstance(accession_value, str) else accession_value

        # Load each image volume and stack along channel dimension
        image_paths = self._parse_image_paths(row["image_paths"])
        volumes: List[torch.Tensor] = []
        for path in image_paths:
            vol = rve.load_sample(path, use_hardware_acceleration=False)
            if not isinstance(vol, torch.Tensor):
                vol = torch.as_tensor(vol)
            # Expect (D, H, W)
            if vol.dim() != 3:
                raise ValueError(f"Expected 3D volume from RVE, got shape {tuple(vol.shape)} for path {path}")
            volumes.append(vol)

        image = torch.stack(volumes, dim=0)  # (C, D, H, W)
        image = self._crop_side(image, self.img_size)
        image = self._pad_along_depth(image, self.num_images)

        batch: Dict[str, Any] = {}
        batch["x"] = image

        # Handle mask if present, else zero mask
        mask_tensor: Optional[torch.Tensor] = None
        if self.has_mask_col:
            mask_path = row["mask_path"]
            if isinstance(mask_path, str) and len(mask_path) > 0 and os.path.exists(mask_path):
                mask_vol = rve.load_sample(mask_path, use_hardware_acceleration=False)
                if not isinstance(mask_vol, torch.Tensor):
                    mask_vol = torch.as_tensor(mask_vol)
                if mask_vol.dim() != 3:
                    raise ValueError(
                        f"Expected 3D mask volume from RVE, got shape {tuple(mask_vol.shape)} for path {mask_path}"
                    )
                mask_vol = mask_vol == 1
                mask_vol = self._pad_along_depth(mask_vol, self.num_images)
                mask_vol = self._crop_side(mask_vol, self.img_size)
                mask_tensor = mask_vol.unsqueeze(0)  # (1, D, H, W)

        if mask_tensor is None:
            # Create a zero mask matching spatial size of x
            _, d, h, w = image.shape
            mask_tensor = torch.zeros((1, d, h, w), dtype=torch.bool)

        batch["mask"] = mask_tensor
        if mask_tensor.sum() > 0:
            batch["image_annotations"] = mask_tensor.float() / mask_tensor.sum()
            batch["has_annotation"] = True
        else:
            batch["image_annotations"] = torch.zeros_like(mask_tensor, dtype=torch.float32)
            batch["has_annotation"] = False

        # Add passthrough columns unchanged
        for col in self.label_columns:
            value = row[col]
            if isinstance(value, str):
                value = ast.literal_eval(value)
            batch[col] = torch.tensor(value)

        # Compute survival labels (y_seq, y_mask) if y and time_at_event are present
        if "y" in row and "time_at_event" in row:
            y_val = int(row["y"])
            time_at_event_val = int(row["time_at_event"])
            y_seq, y_mask = self._compute_survival_labels(y_val, time_at_event_val)
            batch["y_seq"] = torch.tensor(y_seq)
            batch["y_mask"] = torch.tensor(y_mask)

        batch["accession"] = accession
        batch["sample_name"] = accession
        batch["anatomy"] = self.anatomy
        return batch
