from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from totalfm.image_proc import resize_organ_patch, windowing_3ch

# Organ label index to name mapping (TotalSegmentator v2 "total" task).
# Reference: https://github.com/wasserth/TotalSegmentator
INDEX_TO_ORGAN: dict[int, str] = {
    1:   "spleen",
    2:   "kidney_right",
    3:   "kidney_left",
    4:   "gallbladder",
    5:   "liver",
    6:   "stomach",
    7:   "pancreas",
    8:   "adrenal_gland_right",
    9:   "adrenal_gland_left",
    10:  "lung_upper_lobe_left",
    11:  "lung_lower_lobe_left",
    12:  "lung_upper_lobe_right",
    13:  "lung_middle_lobe_right",
    14:  "lung_lower_lobe_right",
    15:  "esophagus",
    16:  "trachea",
    17:  "thyroid_gland",
    18:  "small_bowel",
    19:  "duodenum",
    20:  "colon",
    21:  "urinary_bladder",
    22:  "prostate",
    23:  "kidney_cyst_left",
    24:  "kidney_cyst_right",
    25:  "sacrum",
    26:  "vertebrae_S1",
    27:  "vertebrae_L5",
    28:  "vertebrae_L4",
    29:  "vertebrae_L3",
    30:  "vertebrae_L2",
    31:  "vertebrae_L1",
    32:  "vertebrae_T12",
    33:  "vertebrae_T11",
    34:  "vertebrae_T10",
    35:  "vertebrae_T9",
    36:  "vertebrae_T8",
    37:  "vertebrae_T7",
    38:  "vertebrae_T6",
    39:  "vertebrae_T5",
    40:  "vertebrae_T4",
    41:  "vertebrae_T3",
    42:  "vertebrae_T2",
    43:  "vertebrae_T1",
    44:  "vertebrae_C7",
    45:  "vertebrae_C6",
    46:  "vertebrae_C5",
    47:  "vertebrae_C4",
    48:  "vertebrae_C3",
    49:  "vertebrae_C2",
    50:  "vertebrae_C1",
    51:  "heart",
    52:  "aorta",
    53:  "pulmonary_vein",
    54:  "brachiocephalic_trunk",
    55:  "subclavian_artery_right",
    56:  "subclavian_artery_left",
    57:  "common_carotid_artery_right",
    58:  "common_carotid_artery_left",
    59:  "brachiocephalic_vein_left",
    60:  "brachiocephalic_vein_right",
    61:  "atrial_appendage_left",
    62:  "superior_vena_cava",
    63:  "inferior_vena_cava",
    64:  "portal_vein_and_splenic_vein",
    65:  "iliac_artery_left",
    66:  "iliac_artery_right",
    67:  "iliac_vena_left",
    68:  "iliac_vena_right",
    69:  "humerus_left",
    70:  "humerus_right",
    71:  "scapula_left",
    72:  "scapula_right",
    73:  "clavicula_left",
    74:  "clavicula_right",
    75:  "femur_left",
    76:  "femur_right",
    77:  "hip_left",
    78:  "hip_right",
    79:  "spinal_cord",
    80:  "gluteus_maximus_left",
    81:  "gluteus_maximus_right",
    82:  "gluteus_medius_left",
    83:  "gluteus_medius_right",
    84:  "gluteus_minimus_left",
    85:  "gluteus_minimus_right",
    86:  "autochthon_left",
    87:  "autochthon_right",
    88:  "iliopsoas_left",
    89:  "iliopsoas_right",
    90:  "brain",
    91:  "skull",
    92:  "rib_left_1",
    93:  "rib_left_2",
    94:  "rib_left_3",
    95:  "rib_left_4",
    96:  "rib_left_5",
    97:  "rib_left_6",
    98:  "rib_left_7",
    99:  "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}

ORGAN_TO_INDEX: dict[str, int] = {name: idx for idx, name in INDEX_TO_ORGAN.items()}


def _adjust_axis(
    start: int,
    end: int,
    min_size: int,
    max_size: int,
    limit: int,
) -> Tuple[int, int]:
    """Clamp and expand/shrink one spatial axis to fit size constraints.

    Args:
        start: Bounding-box start coordinate.
        end: Bounding-box end coordinate.
        min_size: Minimum required length along the axis.
        max_size: Maximum allowed length along the axis.
        limit: Image boundary (exclusive) along the axis.

    Returns:
        Adjusted ``(start, end)`` coordinates.
    """
    start = max(0, start)
    end = min(limit, end)
    length = max(1, end - start)

    if limit <= min_size:
        return 0, limit

    if length < min_size:
        needed = min_size - length
        expand_left = needed // 2
        expand_right = needed - expand_left
        start = max(0, start - expand_left)
        end = min(limit, end + expand_right)
        if end - start < min_size:
            deficit = min_size - (end - start)
            if start == 0:
                end = min(limit, end + deficit)
            elif end == limit:
                start = max(0, start - deficit)
            else:
                shift_left = deficit // 2
                start = max(0, start - shift_left)
                end = min(limit, end + (deficit - shift_left))
        return int(start), int(end)

    if length > max_size:
        center = (start + end) / 2
        start = int(round(center - max_size / 2))
        end = start + max_size
        if start < 0:
            start, end = 0, max_size
        if end > limit:
            end = limit
            start = max(0, end - max_size)
        return int(start), int(end)

    return int(start), int(end)


def extract_organ_patch(
    image: np.ndarray,
    segmentation: np.ndarray,
    organ_name: str,
    target_hw: int = 192,
    max_hw: int = 448,
    target_z: int = 32,
) -> Optional[torch.Tensor]:
    """Extract and preprocess a CT organ patch as a model-ready tensor.

    The entire Z range of the organ bounding box is extracted and resized to
    ``target_z`` frames via trilinear interpolation, so one tensor covers the
    whole organ. The XY region is derived from the organ bounding box center,
    cropped to at most ``max_hw`` pixels and resized to ``target_hw``.

    Args:
        image: CT volume array of shape ``(Z, H, W)`` in raw Hounsfield units.
        segmentation: TotalSegmentator integer label array of shape ``(Z, H, W)``.
        organ_name: Name of the target organ (must be a key in
            :data:`ORGAN_TO_INDEX`).
        target_hw: Target height and width after resizing.
        max_hw: Maximum XY crop size before resizing.
        target_z: Target number of depth frames.

    Returns:
        Float tensor of shape ``(3, target_hw, target_hw, target_z)``, or
        ``None`` if the organ is absent from the segmentation.
    """
    label_index = ORGAN_TO_INDEX.get(organ_name)
    if label_index is None:
        return None

    organ_mask = segmentation == label_index
    if not np.any(organ_mask):
        return None

    z_idx, y_idx, x_idx = np.where(organ_mask)
    z_min, z_max = int(z_idx.min()), int(z_idx.max()) + 1
    y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
    x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1

    y_start, y_end = _adjust_axis(y_min, y_max, target_hw, max_hw, image.shape[1])
    x_start, x_end = _adjust_axis(x_min, x_max, target_hw, max_hw, image.shape[2])

    patch = image[z_min:z_max, y_start:y_end, x_start:x_end].astype(np.float32)
    if patch.size == 0:
        return None

    # Apply 3-channel windowing: (Z, H, W) -> (Z, H, W, 3) -> (3, H, W, Z)
    patch_3ch = windowing_3ch(patch)
    patch_t = torch.from_numpy(np.transpose(patch_3ch, (3, 1, 2, 0)).copy())

    # Resize to (3, target_hw, target_hw, target_z) via trilinear interpolation
    return resize_organ_patch(patch_t, target_hw, target_z)
