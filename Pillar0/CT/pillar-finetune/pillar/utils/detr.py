"""
Mostly copy-paste from DETR, modified for 3D. https://github.com/facebookresearch/detr/
"""

from packaging import version
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

if version.parse(torchvision.__version__) < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse("0.7"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(boxes):
    x_min = boxes[..., 0] - boxes[..., 3] / 2
    y_min = boxes[..., 1] - boxes[..., 4] / 2
    z_min = boxes[..., 2] - boxes[..., 5] / 2
    x_max = boxes[..., 0] + boxes[..., 3] / 2
    y_max = boxes[..., 1] + boxes[..., 4] / 2
    z_max = boxes[..., 2] + boxes[..., 5] / 2
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)  # Keep last dim for consistency


def prediction_box_to_3d_mask(pred_boxes, pred_classes, image_size):
    n_samples, num_queries, _ = pred_boxes.shape
    D, H, W = image_size
    mask = torch.zeros((n_samples, D, H, W), dtype=torch.uint8)

    for sample_idx in range(n_samples):
        for query_idx in range(num_queries):
            if pred_classes[sample_idx, query_idx] != 1:
                continue
            x_min, y_min, z_min, x_max, y_max, z_max = box_cxcyczwhd_to_xyzxyz(pred_boxes[sample_idx, query_idx])
            x_min, x_max = max(0, int(x_min * W)), min(W, int(x_max * W))
            y_min, y_max = max(0, int(y_min * H)), min(H, int(y_max * H))
            z_min, z_max = max(0, int(z_min * D)), min(D, int(z_max * D))
            mask[sample_idx, z_min:z_max, y_min:y_max, x_min:x_max] = 1
    return mask


def mask_to_bounding_box(mask):
    """
    Convert a 3D mask to a bounding box.
    mask: Tensor of shape [depth, height, width], where values represent fractional coverage.

    Returns:
        bounding_box: [center_x, center_y, center_z, width, height, depth]
    """
    mask_size = mask.shape
    coords = torch.nonzero(mask > 0)  # Find all nonzero voxel coordinates
    if coords.size(0) == 0:
        return None  # No bounding box if the mask is empty

    # Min and max coordinates along each axis
    z_min, y_min, x_min = coords.min(dim=0).values
    z_max, y_max, x_max = coords.max(dim=0).values

    # Center and dimensions
    center_z = (z_min + z_max) / 2
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2
    depth = z_max - z_min
    height = y_max - y_min
    width = x_max - x_min

    box = torch.tensor([center_x, center_y, center_z, width, height, depth])
    # Normalize [center_x, center_y, center_z, width, height, depth]
    box[:3] /= torch.tensor(mask_size[::-1])  # center coordinates
    box[3:] /= torch.tensor(mask_size[::-1])  # box dimensions

    return box


def generalized_box_iou_3d(boxes1, boxes2):
    """
    Compute Generalized IoU (GIoU) for 3D bounding boxes.
    Params:
        boxes1: Tensor of shape [N, 6], where each box is [cx, cy, cz, w, h, d].
        boxes2: Tensor of shape [M, 6], where each box is [cx, cy, cz, w, h, d].
    Returns:
        Tensor of shape [N, M] containing the GIoU values.
    """
    boxes1_corners = box_cxcyczwhd_to_xyzxyz(boxes1)  # [N, 6]
    boxes2_corners = box_cxcyczwhd_to_xyzxyz(boxes2)  # [M, 6]

    # Compute intersection
    inter_min = torch.max(boxes1_corners[:, None, :3], boxes2_corners[:, :3])  # [N, M, 3]
    inter_max = torch.min(boxes1_corners[:, None, 3:], boxes2_corners[:, 3:])  # [N, M, 3]
    inter_dims = (inter_max - inter_min).clamp(min=0)  # [N, M, 3]
    inter_volume = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]  # [N, M]

    # Compute volume of each box
    def box_volume(corners):
        dims = (corners[:, 3:] - corners[:, :3]).clamp(min=0)
        return dims[:, 0] * dims[:, 1] * dims[:, 2]

    volume1 = box_volume(boxes1_corners)  # [N]
    volume2 = box_volume(boxes2_corners)  # [M]

    # Compute union volume
    union_volume = volume1[:, None] + volume2 - inter_volume  # [N, M]

    # Compute IoU
    iou = inter_volume / union_volume

    # Compute the smallest enclosing box (for GIoU)
    enclosing_min = torch.min(boxes1_corners[:, None, :3], boxes2_corners[:, :3])  # [N, M, 3]
    enclosing_max = torch.max(boxes1_corners[:, None, 3:], boxes2_corners[:, 3:])  # [N, M, 3]
    enclosing_dims = (enclosing_max - enclosing_min).clamp(min=0)  # [N, M, 3]
    enclosing_volume = enclosing_dims[:, :, 0] * enclosing_dims[:, :, 1] * enclosing_dims[:, :, 2]  # [N, M]

    # Compute GIoU
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume

    return giou
