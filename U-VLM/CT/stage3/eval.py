"""
Stage 3 eval：對 val set 做 greedy 生成，算 BLEU-mean（BLEU-1~4 平均，論文 B-mean 定義），
存質化樣本（生成報告 vs 原始 Findings/Impressions，比照論文 Fig.2），並附一個粗略的
關鍵字比對 F1 當 placeholder。

F1 的說明（見 STAGE3_PLAN.md 第 6/8 節）：論文用「CT-RATE 自己的 text classifier」從生成報告
抽 18 類標籤，我們沒有這個工具。這裡先用最陽春的關鍵字比對頂著（把 18 類名稱轉小寫直接找
substring），**只是一個粗略 placeholder，不是正式的 F1 數字**，之後要換成 STAGE3_PLAN.md
建議的 LLM 抽取法才能真的拿來比較模型好壞。

跑法：python3 U-VLM/stage3/eval.py --decoder-checkpoint U-VLM/stage3/checkpoints/best_decoder.pt
"""
import argparse
import csv
import os

import torch
import yaml
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader

from dataset import Stage3Dataset, build_tokenizer
from model import Stage3Model
from train import get_device, move_batch_to_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 18 類病理名稱（跟 U-VLM/stage2/config.yaml 的 labels 順序一致），只用來做粗略關鍵字比對 F1
LABEL_KEYWORDS = [
    "lung nodule", "lung opacity", "pulmonary fibrotic sequela", "atelectasis", "consolidation",
    "lymphadenopathy", "emphysema", "arterial wall calcification", "coronary artery wall calcification",
    "bronchiectasis", "pleural effusion", "peribronchial thickening", "medical material",
    "interlobular septal thickening", "pericardial effusion", "mosaic attenuation pattern",
    "cardiomegaly", "hiatal hernia",
]


@torch.no_grad()
def generate_reports(model, batch, tokenizer, max_new_tokens, device):
    """Greedy batched decoding：整批同步一步步長，遇到 EOS 的樣本繼續跑但輸出會在
    decode_text 那步被截斷，不影響結果，只是多花一點運算。"""
    model.eval()
    features = model.encoder(batch["image"])
    B = batch["image"].shape[0]

    generated = torch.full((B, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)  # BOS=eos_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        report_mask = torch.ones_like(generated)
        logits = model.decoder(features, batch["instruction_ids"], generated, report_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == tokenizer.eos_token_id)
        if finished.all():
            break

    texts = []
    for row in generated[:, 1:]:  # 去掉開頭的 BOS
        ids = row.tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def keyword_label_vector(text):
    text_lower = text.lower()
    return [1 if kw in text_lower else 0 for kw in LABEL_KEYWORDS]


def keyword_f1(pred_texts, ref_texts):
    """粗略 placeholder：兩邊都用關鍵字比對抽出 18 維向量，算 micro F1。"""
    tp = fp = fn = 0
    for pred_text, ref_text in zip(pred_texts, ref_texts):
        pred_vec = keyword_label_vector(pred_text)
        ref_vec = keyword_label_vector(ref_text)
        for p, r in zip(pred_vec, ref_vec):
            if p == 1 and r == 1:
                tp += 1
            elif p == 1 and r == 0:
                fp += 1
            elif p == 0 and r == 1:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def bleu_mean(hypotheses, references):
    scores = []
    for n in (1, 2, 3, 4):
        bleu = BLEU(max_ngram_order=n)
        scores.append(bleu.corpus_score(hypotheses, [references]).score)
    return sum(scores) / len(scores), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--decoder-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_decoder.pt"))
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="預設用 config 裡 Stage 2 的 encoder checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 筆 val（debug 用）")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    tokenizer = build_tokenizer(config["text"]["tokenizer_name"])
    val_ds = Stage3Dataset(config, "val", augment=False, tokenizer=tokenizer)
    if args.limit:
        val_ds.records = val_ds.records[: args.limit]
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False,
                             num_workers=config["train"]["num_workers"])

    model = Stage3Model(config, vocab_size=tokenizer.vocab_size).to(device)
    encoder_ckpt = args.encoder_checkpoint or os.path.join(
        os.path.dirname(os.path.dirname(SCRIPT_DIR)), config["data"]["encoder_checkpoint"]
    )
    model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    model.decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device))
    model.eval()

    all_hyp, all_ref, all_ids = [], [], []
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        hyps = generate_reports(model, batch, tokenizer, args.max_new_tokens, device)
        all_hyp.extend(hyps)
        all_ref.extend(batch["report_text"])
        all_ids.extend(batch["item_id"])
        print(f"generated {len(all_hyp)}/{len(val_ds)}")

    bm, per_n = bleu_mean(all_hyp, all_ref)
    precision, recall, f1 = keyword_f1(all_hyp, all_ref)

    print(f"\nBLEU-1..4: {[round(s, 2) for s in per_n]}")
    print(f"BLEU-mean: {bm:.4f}")
    print(f"Keyword-match F1 (placeholder, 不是正式指標): "
          f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    checkpoint_dir = os.path.dirname(args.decoder_checkpoint)
    out_path = os.path.join(checkpoint_dir, "eval_generated_reports.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "generated_report", "reference_report"])
        for item_id, hyp, ref in zip(all_ids, all_hyp, all_ref):
            writer.writerow([item_id, hyp, ref])
    print(f"\n生成報告 vs 原始報告存到 {out_path}，人工核對生成品質")


if __name__ == "__main__":
    main()
