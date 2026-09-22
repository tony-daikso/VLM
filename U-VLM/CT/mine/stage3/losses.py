"""
標準 autoregressive cross-entropy（論文 Eq.3），只對 report token 這段計算 loss，
instruction/vision token 位置不計入（見 STAGE3_PLAN.md 第 5 節）。
"""
import torch.nn.functional as F


def report_generation_loss(logits, report_ids, report_mask):
    """
    logits: (B, R, vocab_size) —— decoder 對 report 這段每個位置的下一個 token 預測
    report_ids: (B, R) —— [BOS] + text + [EOS] + pad
    report_mask: (B, R) —— 1=真實 token（含 BOS/EOS），0=pad

    position t 的 logits 預測 position t+1 的 token，最後一個位置沒有下一個 token 可預測，捨去。
    """
    pred_logits = logits[:, :-1, :].contiguous()
    target = report_ids[:, 1:].clone()
    target_mask = report_mask[:, 1:]
    target[target_mask == 0] = -100

    vocab_size = pred_logits.shape[-1]
    loss = F.cross_entropy(
        pred_logits.view(-1, vocab_size), target.view(-1), ignore_index=-100
    )
    return loss
