"""
Multi-label BCEWithLogitsLoss，每個類別各自的 pos_weight = min(neg_count / pos_count, cap)
（用 train split 統計出來的類別數算，cap 見 config train.pos_weight_cap，預設 20 倍 -- 避免
osteopenia/goiter 這種極端稀有類別把 pos_weight 推到誇張倍數，導致訓練不穩定，見
STAGE2_PLAN.md 第 4、7 節）。
"""
import torch
import torch.nn as nn


def compute_pos_weight(train_dataset, label_names, cap):
    """train_dataset: Stage2Dataset(split="train")，回傳 (num_labels,) tensor。"""
    pos_counts = torch.zeros(len(label_names))
    for r in train_dataset.records:
        for i, name in enumerate(label_names):
            pos_counts[i] += r["classification_labels"][name]

    n = len(train_dataset.records)
    neg_counts = n - pos_counts
    return (neg_counts / pos_counts.clamp(min=1)).clamp(max=cap)


def build_criterion(pos_weight):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
