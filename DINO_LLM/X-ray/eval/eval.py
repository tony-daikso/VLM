"""
在 validation/test split 上跑 DINO+LLaVA(Llama-3.2-3B) 推論生成報告，算 BLEU/ROUGE +
keyword 臨床有效性 proxy，跟 U-VLM/X-ray/cxr/stage3/eval.py 用同一套指標實作（同樣的
sacrebleu/rouge_score 計算方式、同樣的 26 個 label_group 關鍵字比對邏輯），才能直接對照兩邊
的數字。

生成邏輯照抄 DINO_LLM/prior/inference.py::llava_inference()：固定 prompt
"generate analysis report"、llama3 conversation template、temperature=0（greedy）。

用法（--llava_model 指某一個 epoch 的 checkpoint 資料夾，例如 eval_loss 最低的那個）：
    python3 eval.py \
        --llava_model /datadrive/VLM/DINO_LLM/X-ray/stage_llava_ft/checkpoints/checkpoint-597 \
        --model_base /datadrive/VLM/DINO_LLM/X-ray/models/Llama-3.2-3B \
        --manifest /root/Desktop/VLM/data/X-ray/PadChest-GR/processed/manifest.jsonl \
        --mosaic_dir /datadrive/VLM/DINO_LLM/X-ray/mosaics \
        --split validation \
        --labels_config /root/Desktop/VLM/U-VLM/X-ray/cxr/stage3/config.yaml \
        --out_dir /datadrive/VLM/DINO_LLM/X-ray/eval_results
"""
import argparse
import csv
import json
import os
import sys

import sacrebleu
import torch
import yaml
from PIL import Image
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transformers

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

PROMPT = "generate analysis report"
CONV_MODE = "llama3"


def load_model_for_eval(model_base, checkpoint_dir):
    """
    重寫一個專用的 loader，不用 llava/model/builder.py 通用的 load_pretrained_model()：
    那支函式的 LoRA 分支假設 vocab_size 訓練時被 resize 過一格（給 pad token），會直接
    `vocab_size - 1`，這裡的訓練流程因為 tokenizer 本來就有 pad_token（Llama-3.2 內建
    `<|finetune_right_pad_id|>`），train.py 從沒進到 resize 那條分支，vocab_size 從頭到尾
    跟 base model 一樣，用那個減一邏輯會直接對不上 shape。而且它靠 model_name 字串裡有沒有
    "lora" 來判斷要走哪條路，這裡也懶得硬湊字串，直接照 stage_llava_ft/train_lora.sh 訓練時
    的初始化流程重寫一次比較保險（跟 smoke test 驗證過的流程一致）。
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlavaLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16).cuda()
    model.get_model().initialize_vision_modules(model_args=type("A", (), {
        "vision_tower": "openai/clip-vit-large-patch14-336", "mm_vision_select_layer": -2,
        "pretrain_mm_mlp_adapter": None, "mm_projector_type": "mlp2x_gelu", "mm_patch_merge_type": "flat",
        "mm_vision_select_feature": "patch",
    })())
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device="cuda")
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False

    mm_projector_weights = torch.load(os.path.join(checkpoint_dir, "mm_projector.bin"), map_location="cpu")
    model.get_model().mm_projector.load_state_dict(
        {k.split("mm_projector.")[-1]: v for k, v in mm_projector_weights.items()}, strict=True,
    )
    model.get_model().mm_projector.to(dtype=torch.float16, device="cuda")

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
    model.eval()

    return tokenizer, model, vision_tower.image_processor


def load_manifest(path, split):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


@torch.no_grad()
def generate_report(model, tokenizer, image_processor, mosaic_path, max_new_tokens):
    image = Image.open(mosaic_path).convert("RGB")
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + PROMPT \
        if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN + "\n" + PROMPT

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    output_ids = model.generate(
        input_ids, images=images_tensor, image_sizes=[image.size],
        do_sample=False, temperature=0.0, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def compute_bleu_rouge(generated, references):
    bleu = sacrebleu.corpus_bleu(generated, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        for k in sums:
            sums[k] += scores[k].fmeasure
    n = max(len(generated), 1)
    return bleu.score, {k: v / n for k, v in sums.items()}


def keyword_clinical_proxy(generated, gt_labels, label_names):
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

    results, f1s = {}, []
    for name, c in per_label.items():
        precision = c["tp"] / max(c["tp"] + c["fp"], 1)
        recall = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        results[name] = {"n_pos": c["n_pos"], "precision": precision, "recall": recall, "f1": f1}
        if c["n_pos"] > 0:
            f1s.append(f1)
    macro_f1 = sum(f1s) / max(len(f1s), 1)
    return results, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava_model", required=True, help="LoRA checkpoint dir (has adapter_model.safetensors + mm_projector.bin)")
    ap.add_argument("--model_base", required=True, help="base LLM path (e.g. .../models/Llama-3.2-3B)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mosaic_dir", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--labels_config", required=True, help="U-VLM stage3 config.yaml, reuse its `labels` list")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.labels_config) as f:
        label_names = yaml.safe_load(f)["labels"]

    tokenizer, model, image_processor = load_model_for_eval(args.model_base, args.llava_model)

    rows = load_manifest(args.manifest, args.split)
    generated, references, study_ids, gt_labels = [], [], [], []
    for i, row in enumerate(rows):
        mosaic_path = os.path.join(args.mosaic_dir, args.split, f"{row['study_id']}.png")
        if not os.path.exists(mosaic_path):
            continue
        text = generate_report(model, tokenizer, image_processor, mosaic_path, args.max_new_tokens)
        generated.append(text)
        references.append(row.get("report_text") or "")
        study_ids.append(row["study_id"])
        gt_labels.append(row.get("classification_labels", {}))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}")

    bleu, rouge_avg = compute_bleu_rouge(generated, references)
    print(f"\n=== {args.split} 生成品質(n={len(generated)}) ===")
    print(f"BLEU: {bleu:.2f}")
    print(f"ROUGE-1/2/L (F1): {rouge_avg['rouge1']:.4f} / {rouge_avg['rouge2']:.4f} / {rouge_avg['rougeL']:.4f}")

    per_label, macro_f1 = keyword_clinical_proxy(generated, gt_labels, label_names)
    print("\n=== keyword 臨床有效性 proxy(粗略,同 U-VLM stage3 eval.py 的定義) ===")
    for name, m in per_label.items():
        print(f"  {name}: n_pos={m['n_pos']} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
    print(f"\nmacro F1 (over labels with n_pos>0): {macro_f1:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"eval_generations_{args.split}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "generated", "reference"])
        for sid, gen, ref in zip(study_ids, generated, references):
            w.writerow([sid, gen, ref])
    with open(os.path.join(args.out_dir, f"eval_keyword_proxy_{args.split}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_pos", "precision", "recall", "f1"])
        for name, m in per_label.items():
            w.writerow([name, m["n_pos"], round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4)])
    print(f"\n結果存到 {args.out_dir}/eval_generations_{args.split}.csv, eval_keyword_proxy_{args.split}.csv")


if __name__ == "__main__":
    main()
