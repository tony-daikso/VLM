"""
單一 lesion head 的 loss：Dice + BCE（見 STAGE1_PLAN.md 第 5 節）。

跟 CT 版不同，這裡只有一個資料來源、一個 head，每個 batch 裡所有樣本都對這個 loss 有貢獻
（normal study 的 target 就是全 0 mask，一樣正常參與 loss，不需要 CT 版那套「valid_mask」
多來源 masking 機制）。
"""
import torch
import torch.nn.functional as F

EPS = 1e-6


def dice_loss_binary(logits, target):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + EPS) / (union + EPS)
    return 1 - dice.mean()


def bce_loss_binary(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_losses(outputs, batch, config):
    dice_w = config["loss"]["dice_weight"]
    bce_w = config["loss"]["bce_weight"]

    dice = dice_loss_binary(outputs["lesion_logits"], batch["lesion_mask"])
    bce = bce_loss_binary(outputs["lesion_logits"], batch["lesion_mask"])
    total = dice_w * dice + bce_w * bce

    return {"total": total, "dice": dice.detach(), "bce": bce.detach()}
