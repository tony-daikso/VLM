import torch

import torch


def compute_dice(mask1, mask2, epsilon=1e-6):
    """
    Computes the Dice coefficient between two binary masks in a batched manner.
    Assumes mask1 and mask2 are tensors of shape (B, D, H, W).
    """
    intersection = torch.sum(mask1 * mask2, dim=(1, 2, 3))  # Sum over spatial dims
    union = torch.sum(mask1, dim=(1, 2, 3)) + torch.sum(mask2, dim=(1, 2, 3))

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice  # Returns a (B,) tensor


def compute_dice_foreground(mask1, mask2, epsilon=1e-6):
    """
    Computes the Dice coefficient only on the foreground,
    i.e., only in the voxels where the ground truth (mask2) is positive.
    Works in a batched manner.
    Dimension is: (BS, D, H, W)
    """
    roi = mask2.bool()  # Region of interest: foreground of ground truth

    pred_foreground = torch.where(roi, mask1, torch.tensor(0.0))  # Zero out non-foreground
    gt_foreground = torch.where(roi, mask2, torch.tensor(0.0))

    intersection = torch.sum(pred_foreground * gt_foreground, dim=(1, 2, 3))
    union = torch.sum(pred_foreground, dim=(1, 2, 3)) + torch.sum(gt_foreground, dim=(1, 2, 3))

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice  # Returns a (B,) tensor


def compute_iou(mask1, mask2, epsilon=1e-6):
    """
    Computes the Intersection over Union (IoU) between two binary masks in a batched manner.
    Dimension is: (BS, D, H, W)
    """
    intersection = torch.sum(mask1 * mask2, dim=(1, 2, 3))
    union = torch.sum(mask1, dim=(1, 2, 3)) + torch.sum(mask2, dim=(1, 2, 3)) - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    return iou  # Returns a (B,) tensor
