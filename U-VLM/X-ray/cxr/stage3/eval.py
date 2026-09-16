"""
跑 val/test set，生成報告，輸出 BLEU/ROUGE + keyword 臨床有效性 proxy 指標（見 STAGE3_PLAN.md
第 3、5 節）。

跑法：python3 U-VLM/X-ray/cxr/stage3/eval.py --checkpoint U-VLM/X-ray/cxr/stage3/checkpoints/best_model.pt
"""
import argparse
import csv
import os

import sacrebleu
import torch
import yaml
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from dataset import Stage3Dataset, make_collate_fn
from model import Stage3Model
from train import build_tokenizer, get_device, resolve_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def generate_reports(model, loader, tokenizer, device, max_new_tokens, num_beams):
    generated, references, study_ids, gt_labels = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        output_ids = model.generate(image, tokenizer, max_new_tokens, num_beams)
        for i in range(image.shape[0]):
            text = tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
            generated.append(text)
        references.extend(batch["report_text"])
        study_ids.extend(batch["study_id"])
        gt_labels.extend(batch["classification_labels"])
    return generated, references, study_ids, gt_labels


def compute_bleu_rouge(generated, references):
    bleu = sacrebleu.corpus_bleu(generated, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        for k in rouge_sums:
            rouge_sums[k] += scores[k].fmeasure
    n = max(len(generated), 1)
    rouge_avg = {k: v / n for k, v in rouge_sums.items()}
    return bleu.score, rouge_avg


def keyword_clinical_proxy(generated, gt_labels, label_names):
    """粗略 proxy：檢查生成文字有沒有出現各 label_group 名稱的關鍵字，跟真實
    classification_labels 比對，算每個 label 的 TP/FP/FN，再算 micro/macro precision/recall/F1
    （見 STAGE3_PLAN.md 第 5 節：這不是精確指標，只是比 BLEU 更貼近「有沒有講對病灶」）。"""
    per_label = {name: {"tp": 0, "fp": 0, "fn": 0, "n_pos": 0} for name in label_names}
    for text, labels in zip(generated, gt_labels):
        text_lower = text.lower()
        for name in label_names:
            mentioned = name.lower() in text_lower
            is_positive = bool(labels.get(name, 0))
            if is_positive:
                per_label[name]["n_pos"] += 1
            if mentioned and is_positive:
                per_label[name]["tp"] += 1
            elif mentioned and not is_positive:
                per_label[name]["fp"] += 1
            elif not mentioned and is_positive:
                per_label[name]["fn"] += 1

    results = {}
    f1s = []
    for name, c in per_label.items():
        precision = c["tp"] / max(c["tp"] + c["fp"], 1)
        recall = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        results[name] = {"n_pos": c["n_pos"], "precision": precision, "recall": recall, "f1": f1}
        if c["n_pos"] > 0:
            f1s.append(f1)
    macro_f1 = sum(f1s) / max(len(f1s), 1)
    return results, macro_f1


def write_generations_csv(path, study_ids, generated, references):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "generated", "reference"])
        for sid, gen, ref in zip(study_ids, generated, references):
            writer.writerow([sid, gen, ref])


def write_keyword_csv(path, per_label):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n_pos", "precision", "recall", "f1"])
        for name, m in per_label.items():
            writer.writerow([name, m["n_pos"], round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    tokenizer = build_tokenizer(config["model"]["lm_name"])
    model = Stage3Model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = Stage3Dataset(config, args.split, tokenizer)
    loader = DataLoader(
        ds, batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"]["num_workers"], collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    eval_cfg = config["eval"]
    generated, references, study_ids, gt_labels = generate_reports(
        model, loader, tokenizer, device, eval_cfg["max_new_tokens"], eval_cfg["num_beams"]
    )

    bleu, rouge_avg = compute_bleu_rouge(generated, references)
    print(f"\n=== {args.split} 生成品質（n={len(generated)}）===")
    print(f"BLEU: {bleu:.2f}")
    print(f"ROUGE-1/2/L (F1): {rouge_avg['rouge1']:.4f} / {rouge_avg['rouge2']:.4f} / {rouge_avg['rougeL']:.4f}")

    per_label, macro_f1 = keyword_clinical_proxy(generated, gt_labels, config["labels"])
    print(f"\n=== keyword 臨床有效性 proxy（粗略，見 STAGE3_PLAN.md 第 5 節）===")
    for name, m in per_label.items():
        print(f"  {name}: n_pos={m['n_pos']} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
    print(f"\nmacro F1 (over labels with n_pos>0): {macro_f1:.4f}")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    write_generations_csv(
        os.path.join(checkpoint_dir, f"eval_generations_{args.split}.csv"), study_ids, generated, references
    )
    write_keyword_csv(os.path.join(checkpoint_dir, f"eval_keyword_proxy_{args.split}.csv"), per_label)
    print(f"\n生成結果存到 {checkpoint_dir}/eval_generations_{args.split}.csv")
    print(f"keyword proxy 存到 {checkpoint_dir}/eval_keyword_proxy_{args.split}.csv")


if __name__ == "__main__":
    main()
