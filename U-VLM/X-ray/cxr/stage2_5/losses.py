"""
CenterNet 風格的三個 loss：modified focal loss（heatmap）+ masked L1（size、offset），
見 STAGE2_5_PLAN.md 第 5 節。
"""
import torch


def modified_focal_loss(pred, target, alpha, beta):
    """pred/target: (B, C, F, F)，target 是 encode_targets 畫出來的 gaussian heatmap（peak=1，
    其餘介於 0~1）。像素值等於 1 的當正樣本，其餘（包含 gaussian 邊緣的 soft 值）當負樣本但用
    (1-target)^beta 降權，離 GT 中心越近懲罰越輕 -- 這是 CenterNet 論文的標準寫法。"""
    eps = 1e-6
    pred = pred.clamp(min=eps, max=1 - eps)
    pos_mask = (target == 1).float()
    neg_mask = 1 - pos_mask

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_mask

    num_pos = pos_mask.sum()
    loss = -(pos_loss.sum() + neg_loss.sum())
    if num_pos > 0:
        loss = loss / num_pos
    return loss


def masked_l1_loss(pred, target, reg_mask):
    """pred/target: (B, 2, F, F)，reg_mask: (B, 1, F, F)，只在 box 中心點所在的 pixel 算 L1。"""
    mask = reg_mask.expand_as(pred)
    num_valid = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * mask).sum() / num_valid


def compute_losses(outputs, batch, config):
    loss_cfg = config["loss"]
    heatmap_loss = modified_focal_loss(
        outputs["heatmap"], batch["heatmap"], loss_cfg["focal_alpha"], loss_cfg["focal_beta"]
    )
    size_loss = masked_l1_loss(outputs["size"], batch["size_target"], batch["reg_mask"])
    offset_loss = masked_l1_loss(outputs["offset"], batch["offset_target"], batch["reg_mask"])

    total = heatmap_loss + loss_cfg["size_weight"] * size_loss + loss_cfg["offset_weight"] * offset_loss
    return {
        "total": total,
        "heatmap": heatmap_loss.detach(),
        "size": size_loss.detach(),
        "offset": offset_loss.detach(),
    }
