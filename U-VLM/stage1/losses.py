"""
三個 head 各自的 loss（見 STAGE1_PLAN.md 第 5 節）：
  Lesion（binary）：Dice + BCE
  Organ（118 類）：Dice(multi-class) + CE
  Vessel（2 個獨立 binary channel，artery/vein）：Dice + BCE，兩者算完平均

Masked 合併：一個 batch 裡同時有 ctrate 樣本（有 lesion/organ 標註、沒有 vessel 標註）跟
hipas 樣本（有 vessel 標註、沒有 lesion/organ 標註），每個 head 的 loss 只在「該 head 實際
有標註的樣本」上算、除以那些樣本的數量做平均 —— 不能直接對整個 batch 平均，否則兩種來源
比例不同時會互相稀釋（例如 batch 裡 ctrate 樣本多，vessel loss 除以 batch size 會被稀釋成
一個很小的數字，即使 hipas 樣本自己的 vessel loss 很大）。
"""
import torch
import torch.nn.functional as F

EPS = 1e-6


def dice_loss_binary(logits, target, valid_mask):
    """logits/target: (B, C, H, W)，valid_mask: (B,) bool -> 只在 valid 樣本上算，
    每個樣本每個 channel 各自算一個 dice 再平均。"""
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0

    probs = torch.sigmoid(logits)[valid_mask]
    target = target[valid_mask]

    dims = (0, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + EPS) / (union + EPS)
    return 1 - dice.mean()


def bce_loss_binary(logits, target, valid_mask):
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid_mask], target[valid_mask])


def dice_loss_multiclass(logits, target, valid_mask, num_classes):
    """logits: (B, K, H, W), target: (B, H, W) int64 -> 只在 valid 樣本上算，
    每個 class 各自算一個 dice 再對所有 class 平均（macro）。"""
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0

    probs = torch.softmax(logits[valid_mask], dim=1)
    target_onehot = F.one_hot(target[valid_mask], num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dim=dims)
    union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2 * intersection + EPS) / (union + EPS)
    return 1 - dice.mean()


def ce_loss_multiclass(logits, target, valid_mask):
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid_mask], target[valid_mask])


def compute_losses(outputs, batch, config):
    """回傳每個 head 的 loss 值（dict）以及加總的 total loss，供 train.py 做 backward + logging。"""
    valid_lesion_organ = batch["valid_lesion_organ"]
    valid_vessel = batch["valid_vessel"]

    dice_w = config["loss"]["dice_weight"]
    ce_w = config["loss"]["bce_or_ce_weight"]

    lesion_dice = dice_loss_binary(outputs["lesion_logits"], batch["lesion_mask"], valid_lesion_organ)
    lesion_bce = bce_loss_binary(outputs["lesion_logits"], batch["lesion_mask"], valid_lesion_organ)
    lesion_loss = dice_w * lesion_dice + ce_w * lesion_bce

    num_organ_classes = config["organ"]["num_classes"]
    organ_dice = dice_loss_multiclass(outputs["organ_logits"], batch["organ_mask"], valid_lesion_organ, num_organ_classes)
    organ_ce = ce_loss_multiclass(outputs["organ_logits"], batch["organ_mask"], valid_lesion_organ)
    organ_loss = dice_w * organ_dice + ce_w * organ_ce

    artery_logits = outputs["vessel_logits"][:, 0:1]
    vein_logits = outputs["vessel_logits"][:, 1:2]
    artery_target = batch["vessel_mask"][:, 0:1]
    vein_target = batch["vessel_mask"][:, 1:2]

    artery_dice = dice_loss_binary(artery_logits, artery_target, valid_vessel)
    artery_bce = bce_loss_binary(artery_logits, artery_target, valid_vessel)
    vein_dice = dice_loss_binary(vein_logits, vein_target, valid_vessel)
    vein_bce = bce_loss_binary(vein_logits, vein_target, valid_vessel)
    vessel_loss = dice_w * (artery_dice + vein_dice) / 2 + ce_w * (artery_bce + vein_bce) / 2

    total_loss = lesion_loss + organ_loss + vessel_loss

    return {
        "total": total_loss,
        "lesion": lesion_loss.detach(),
        "organ": organ_loss.detach(),
        "vessel": vessel_loss.detach(),
    }
