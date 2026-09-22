from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def windowing(
    img: np.ndarray,
    wl: int = 0,
    ww: int = 400,
    mode: str = "float32",
) -> np.ndarray:
    """Apply CT windowing (intensity clipping and normalisation).

    Args:
        img: Input CT array.
        wl: Window level (center).
        ww: Window width.
        mode: Output dtype. One of ``"float32"`` (0–1) or ``"uint8"`` (0–255).

    Returns:
        Windowed array with the same spatial shape as ``img``.
    """
    floor = wl - ww // 2
    ceil = wl + ww // 2
    img = np.clip(img, floor, ceil)
    if mode == "float32":
        return ((img - floor) / (ceil - floor)).astype(np.float32)
    if mode == "uint8":
        return (((img - floor) / (ceil - floor)) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported mode: {mode}")


def windowing_3ch(img: np.ndarray) -> np.ndarray:
    """Apply three-channel CT windowing (lung / soft tissue / bone).

    Args:
        img: Input CT array of shape ``(Z, H, W)``.

    Returns:
        Three-channel array of shape ``(Z, H, W, 3)`` with float32 values in
        the range [0, 1].
    """
    ch_lung = windowing(img, wl=-600, ww=1500, mode="float32")
    ch_soft = windowing(img, wl=40,   ww=400,  mode="float32")
    ch_bone = windowing(img, wl=300,  ww=1500, mode="float32")
    return np.stack([ch_lung, ch_soft, ch_bone], axis=-1)


def resize_organ_patch(
    patch: torch.Tensor,
    target_hw: int,
    target_z: int,
) -> torch.Tensor:
    """Resize an organ patch tensor to ``(C, target_hw, target_hw, target_z)``.

    Uses trilinear interpolation so that the full spatial extent of the organ
    is covered by a single model-ready volume.

    Args:
        patch: Input tensor of shape ``(C, H, W, Z)``.
        target_hw: Target height and width.
        target_z: Target depth (number of frames).

    Returns:
        Resized tensor of shape ``(C, target_hw, target_hw, target_z)``.
    """
    # (C, H, W, Z) -> (1, C, Z, H, W) for F.interpolate trilinear
    vol = patch.permute(0, 3, 1, 2).unsqueeze(0).float()
    vol = F.interpolate(
        vol,
        size=(target_z, target_hw, target_hw),
        mode="trilinear",
        align_corners=False,
    )
    # (1, C, target_z, target_hw, target_hw) -> (C, target_hw, target_hw, target_z)
    return vol.squeeze(0).permute(0, 2, 3, 1)


def load_nifti_and_resample(
    nifti_path: str,
    interpolator: str = "linear",
) -> np.ndarray:
    """Load a NIfTI file and resample it to a fixed minimum spacing.

    Resamples to ``(max(0.6, sx), max(0.6, sy), max(3.0, sz))`` mm so that
    the input resolution never exceeds the training resolution. Higher-resolution
    images are downsampled; lower-resolution images are left unchanged.

    Args:
        nifti_path: Path to the NIfTI file.
        interpolator: Interpolation method. Use ``"linear"`` for CT images and
            ``"nearest"`` for segmentation masks.

    Returns:
        Resampled array of shape ``(Z, H, W)``.
    """
    image = sitk.ReadImage(nifti_path)
    orig_spacing = np.array(image.GetSpacing(), dtype=float)   # (sx, sy, sz)
    orig_size = np.array(image.GetSize(), dtype=float)         # (nx, ny, nz)

    new_spacing = np.array([
        max(0.6, orig_spacing[0]),
        max(0.6, orig_spacing[1]),
        max(3.0, orig_spacing[2]),
    ])
    new_size = tuple(int(round(s)) for s in (orig_size * orig_spacing / new_spacing))

    if interpolator == "linear":
        sitk_interp = sitk.sitkLinear
    elif interpolator == "nearest":
        sitk_interp = sitk.sitkNearestNeighbor
    else:
        raise ValueError(f"Unsupported interpolator: '{interpolator}'")

    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk_interp,
        image.GetOrigin(),
        tuple(new_spacing),
        image.GetDirection(),
        -1000,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled)  # (Z, H, W)
