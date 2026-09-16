"""
跑 val set，輸出 18 類各自的 AUROC/AUPRC + val 陽性樣本數（見 STAGE2_PLAN.md 第 6 節：
陽性樣本數一定要跟指標一起報告，不能只看數字，尤其 Hiatal hernia 這種個位數陽性的類別）。

另外對每個陽性樣本數夠多的類別，把預測機率最低的幾個「假陰性」樣本連同報告全文一起 dump 出來，
方便人工核對是不是 STAGE2_PLAN.md 第 1.2 節說的「切片本身就看不出這個標籤」，而不是模型學不好。

跑法：python3 U-VLM/stage2/eval.py \
        --encoder-checkpoint U-VLM/stage2/checkpoints/best_encoder.pt \
        --head-checkpoint U-VLM/stage2/checkpoints/best_head.pt
"""
import argparse
import csv
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from dataset import Stage2Dataset
from model import Stage2Model
from train import compute_label_metrics, get_device, move_batch_to_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 這些類別 val 陽性樣本太少（見 STAGE2_PLAN.md 1.1 節），false negative dump 沒有意義，跳過
MIN_POS_FOR_FN_DUMP = 5
NUM_FN_PER_LABEL = 5
REPORT_TEXT_MAX_CHARS = 500


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_probs, all_labels, all_ids, all_reports = [], [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["image"])
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_ids.extend(batch["item_id"])
        all_reports.extend(batch["report_text"])

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_ids,
        all_reports,
    )


def f1_optimal_threshold(y_true, y_prob):
    """回傳 (threshold, f1)。precision_recall_curve 的 thresholds 比 precision/recall 少一個，
    對齊時去掉最後一個 precision/recall（對應 threshold=inf，沒有對應 threshold 值）。"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def write_per_label_csv(path, label_metrics, thresholds):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n_pos_val", "auroc", "auprc", "f1_optimal_threshold", "f1_at_threshold"])
        for name, m in label_metrics.items():
            th, f1 = thresholds.get(name, (None, None))
            writer.writerow([
                name, m["n_pos"],
                "" if m["auroc"] is None else round(m["auroc"], 4),
                "" if m["auprc"] is None else round(m["auprc"], 4),
                "" if th is None else round(th, 4),
                "" if f1 is None else round(f1, 4),
            ])


def dump_false_negatives(path, label_names, probs, labels, ids, reports):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "item_id", "predicted_prob", "report_text"])
        for i, name in enumerate(label_names):
            pos_mask = labels[:, i] > 0.5
            n_pos = int(pos_mask.sum())
            if n_pos < MIN_POS_FOR_FN_DUMP:
                continue
            pos_indices = np.where(pos_mask)[0]
            pos_probs = probs[pos_indices, i]
            order = np.argsort(pos_probs)  # 機率由低到高，最低的就是最可能被漏掉的假陰性
            for idx in pos_indices[order[:NUM_FN_PER_LABEL]]:
                report = reports[idx][:REPORT_TEXT_MAX_CHARS]
                writer.writerow([name, ids[idx], round(float(probs[idx, i]), 4), report])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--encoder-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_encoder.pt"))
    parser.add_argument("--head-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_head.pt"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    model = Stage2Model(config).to(device)
    model.encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
    model.head.load_state_dict(torch.load(args.head_checkpoint, map_location=device))
    model.eval()

    label_names = config["labels"]
    val_ds = Stage2Dataset(config, "val", augment=False)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False,
                             num_workers=config["train"]["num_workers"])

    probs, labels, ids, reports = collect_predictions(model, val_loader, device)
    label_metrics = compute_label_metrics(probs, labels, label_names)

    print("\n=== Val AUROC / AUPRC（每類都附上 val 陽性樣本數，樣本數太少的類別數字不可靠）===")
    thresholds = {}
    for i, name in enumerate(label_names):
        m = label_metrics[name]
        if m["auroc"] is None:
            print(f"  {name}: n_pos={m['n_pos']} -> val 裡只有單一 class，AUROC/AUPRC 無法計算")
            continue
        th, f1 = f1_optimal_threshold(labels[:, i], probs[:, i])
        thresholds[name] = (th, f1)
        print(f"  {name}: n_pos={m['n_pos']} auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} "
              f"f1_optimal_threshold={th:.3f} (f1={f1:.4f})")

    aurocs = [m["auroc"] for m in label_metrics.values() if m["auroc"] is not None]
    auprcs = [m["auprc"] for m in label_metrics.values() if m["auprc"] is not None]
    print(f"\nmacro AUROC (over {len(aurocs)}/{len(label_names)} 類可計算的): "
          f"{np.mean(aurocs):.4f}" if aurocs else "\nmacro AUROC: 無可計算的類別")
    print(f"macro AUPRC (over {len(auprcs)}/{len(label_names)} 類可計算的): "
          f"{np.mean(auprcs):.4f}" if auprcs else "macro AUPRC: 無可計算的類別")

    checkpoint_dir = os.path.dirname(args.encoder_checkpoint)
    write_per_label_csv(os.path.join(checkpoint_dir, "eval_per_label.csv"), label_metrics, thresholds)
    dump_false_negatives(
        os.path.join(checkpoint_dir, "eval_false_negatives.csv"), label_names, probs, labels, ids, reports
    )
    print(f"\n每類指標存到 {checkpoint_dir}/eval_per_label.csv")
    print(f"假陰性樣本（連同報告全文）存到 {checkpoint_dir}/eval_false_negatives.csv，"
          f"人工核對是否為 STAGE2_PLAN.md 第 1.2 節說的切片本身看不出來的情況")


if __name__ == "__main__":
    main()
