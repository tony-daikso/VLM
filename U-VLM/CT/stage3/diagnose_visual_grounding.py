"""
診斷 Stage 3 decoder 是不是真的有用到視覺特徵，還是只是在背 CT-RATE 報告的模板句型
（見跟使用者的討論：BLEU/F1 數字好看，不代表 grounding 是真的，CT-RATE 報告本身高度模板化，
3,042 筆對 43M 參數 decoder 來說可能不夠讓它學會「看圖說話」而不是「背模板」）。

三個檢查，都用同一批抽樣的 val 樣本：

1. **輸出多樣性**：生成報告彼此之間的重複程度，跟真實報告彼此之間的重複程度比——如果生成報告
   比真實報告彼此更像，代表模型可能收斂到少數幾種模板，不太看圖
2. **換圖敏感度**（最直接的因果檢驗）：同一個樣本，正確配對的圖 vs 隨機換成別的樣本的圖，
   分別生成報告，看輸出差多少——如果換圖前後輸出幾乎一樣，代表模型根本沒在用圖
3. **關鍵字標籤對照 ground truth**：生成報告用關鍵字抽出 18 維向量，跟該樣本真實的
   `classification_labels`（不是跟參考報告文字，是跟結構化 ground truth）比對，正確配對 vs
   換圖配對的 F1 有沒有差——有差才代表圖真的有影響生成內容

跑法：python3 U-VLM/stage3/diagnose_visual_grounding.py --n-samples 30
"""
import argparse
import json
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import Stage3Dataset, build_tokenizer
from eval import LABEL_KEYWORDS, generate_reports, keyword_label_vector
from model import Stage3Model
from train import get_device, move_batch_to_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def token_set(text):
    return set(text.lower().split())


def jaccard(a, b):
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def avg_pairwise_jaccard(texts):
    n = len(texts)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard(texts[i], texts[j])
            count += 1
    return total / count


def label_f1(pred_vecs, true_vecs):
    tp = fp = fn = 0
    for pred, true in zip(pred_vecs, true_vecs):
        for p, t in zip(pred, true):
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--decoder-checkpoint",
                         default=os.path.join(SCRIPT_DIR, "checkpoints", "best_decoder.pt"))
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    tokenizer = build_tokenizer(config["text"]["tokenizer_name"])
    val_ds = Stage3Dataset(config, "val", augment=False, tokenizer=tokenizer)

    random.seed(args.seed)
    n = min(args.n_samples, len(val_ds))
    indices = random.sample(range(len(val_ds)), n)
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=n, shuffle=False, num_workers=2)

    model = Stage3Model(config, vocab_size=tokenizer.vocab_size).to(device)
    encoder_ckpt = args.encoder_checkpoint or os.path.join(ROOT_DIR, config["data"]["encoder_checkpoint"])
    model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    model.decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device))
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    # 找 18 類病理名稱：跟 eval.py 的 LABEL_KEYWORDS 對應到 manifest 的 classification_labels
    # 的實際 key（title case，例如 "lung nodule" -> "Lung nodule"）
    sample_record = val_ds.records[indices[0]]
    label_keys = list(sample_record["classification_labels"].keys())
    key_lookup = {k.lower(): k for k in label_keys}
    label_names = [key_lookup[kw] for kw in LABEL_KEYWORDS if kw in key_lookup]
    print(f"對照 {len(label_names)}/{len(LABEL_KEYWORDS)} 類（keyword 對不上 manifest key 的略過）")

    ground_truth = []
    for idx in indices:
        rec = val_ds.records[idx]
        ground_truth.append([int(rec["classification_labels"][name]) for name in label_names])

    # 1) 正確配對：正常生成
    print("\n[1/2] 用正確配對的圖生成...")
    gen_correct = generate_reports(model, batch, tokenizer, args.max_new_tokens, device)

    # 2) 換圖配對：image 打亂成 derangement（沒有任何位置對到自己），instruction/其他不變
    perm = list(range(n))
    random.shuffle(perm)
    while any(i == p for i, p in enumerate(perm)):
        random.shuffle(perm)
    swapped_batch = dict(batch)
    swapped_batch["image"] = batch["image"][perm]
    print("[2/2] 用換掉的圖生成...")
    gen_swapped = generate_reports(model, swapped_batch, tokenizer, args.max_new_tokens, device)

    reference_texts = batch["report_text"]

    # 檢查 1：輸出多樣性
    div_generated = avg_pairwise_jaccard(gen_correct)
    div_reference = avg_pairwise_jaccard(reference_texts)

    # 檢查 2：換圖敏感度（生成內容本身的差異）
    per_sample_overlap = [jaccard(a, b) for a, b in zip(gen_correct, gen_swapped)]
    avg_self_overlap = sum(per_sample_overlap) / len(per_sample_overlap)

    # 檢查 3：關鍵字標籤對 ground truth 的 F1，正確配對 vs 換圖配對
    correct_pred_vecs = [keyword_label_vector(t) for t in gen_correct]
    correct_pred_vecs = [[v[i] for i, kw in enumerate(LABEL_KEYWORDS) if kw in key_lookup] for v in correct_pred_vecs]
    swapped_pred_vecs = [keyword_label_vector(t) for t in gen_swapped]
    swapped_pred_vecs = [[v[i] for i, kw in enumerate(LABEL_KEYWORDS) if kw in key_lookup] for v in swapped_pred_vecs]

    p_correct, r_correct, f1_correct = label_f1(correct_pred_vecs, ground_truth)
    p_swapped, r_swapped, f1_swapped = label_f1(swapped_pred_vecs, ground_truth)

    print("\n" + "=" * 60)
    print(f"樣本數：{n}")
    print(f"\n[檢查 1：輸出多樣性（0=完全重複，1=完全不重複，用 unigram Jaccard 距離的反面算）]")
    print(f"  生成報告彼此的重複度（1-avg_jaccard）: {1 - div_generated:.4f}")
    print(f"  真實報告彼此的重複度（1-avg_jaccard）: {1 - div_reference:.4f}")
    print(f"  -> {'生成報告明顯比真實報告更重複，疑似模板收斂' if div_generated > div_reference + 0.1 else '重複程度跟真實報告差不多，沒有明顯模板收斂的訊號'}")

    print(f"\n[檢查 2：換圖敏感度]")
    print(f"  正確圖 vs 換掉的圖，生成內容的平均重疊度（1=完全一樣）: {avg_self_overlap:.4f}")
    print(f"  -> {'換圖幾乎沒差，模型可能沒有真的在用視覺特徵' if avg_self_overlap > 0.8 else '換圖後內容有明顯變化，模型有在反應圖片輸入'}")

    print(f"\n[檢查 3：關鍵字標籤 vs ground truth classification_labels 的 F1]")
    print(f"  正確配對: precision={p_correct:.4f} recall={r_correct:.4f} f1={f1_correct:.4f}")
    print(f"  換圖配對: precision={p_swapped:.4f} recall={r_swapped:.4f} f1={f1_swapped:.4f}")
    print(f"  -> {'正確配對明顯比換圖配對好，圖的內容確實影響了生成的病理描述' if f1_correct > f1_swapped + 0.05 else '正確配對沒有明顯優於換圖配對，視覺 grounding 存疑'}")
    print("=" * 60)

    out_path = os.path.join(os.path.dirname(args.decoder_checkpoint), "diagnose_visual_grounding.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_samples": n,
            "diversity_generated": div_generated,
            "diversity_reference": div_reference,
            "avg_self_overlap_correct_vs_swapped_image": avg_self_overlap,
            "label_f1_correct_pairing": {"precision": p_correct, "recall": r_correct, "f1": f1_correct},
            "label_f1_swapped_pairing": {"precision": p_swapped, "recall": r_swapped, "f1": f1_swapped},
        }, f, indent=2)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
