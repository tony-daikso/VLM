"""HU window -> nnU-Net input range. Works on torch tensors (GPU) or numpy arrays.

Order of operations (spec 4.3): all spatial ops happen in HU first; windowing is the last step.
"""
from __future__ import annotations

import numpy as np
import torch


def _clip(x, lo, hi):
    return torch.clamp(x, lo, hi) if isinstance(x, torch.Tensor) else np.clip(x, lo, hi)


def apply_window(hu, win: dict, norm: dict):
    """win: {type: baseline} or {level, width}. norm: nnU-Net foreground intensity properties."""
    p005, p995 = float(norm["percentile_00_5"]), float(norm["percentile_99_5"])
    mean, std = float(norm["mean"]), max(float(norm["std"]), 1e-8)
    if isinstance(hu, np.ndarray):
        hu = hu.astype(np.float32)
    else:
        hu = hu.float()
    if win.get("type") == "baseline":
        x = _clip(hu, p005, p995)                      # identical to nnU-Net CTNormalization
    else:
        lo = win["level"] - win["width"] / 2
        hi = win["level"] + win["width"] / 2
        x = _clip((hu - lo) / (hi - lo), 0.0, 1.0)     # window -> [0, 1]
        x = p005 + x * (p995 - p005)                   # stretch back to nnU-Net's clip range
    return (x - mean) / std
