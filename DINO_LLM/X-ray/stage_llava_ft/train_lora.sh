#!/bin/bash
# 對 Llama-3.2-3B + LLaVA 做單階段 LoRA 微調：mm_projector 從零學、LLM 用 LoRA、
# CLIP vision tower 全程凍結（跟學長 inference.py 用未改動的 LLaVA CLIP tower 一致）。
#
# 跳過 LLaVA 官方兩階段 pretrain（stage1 projector-only pretrain 通常要 558K 圖文對，
# 這裡任務單一（固定 prompt 的報告生成）、資料量只有 3,185 筆，兩階段跑法不划算，也沒有對應
# 的 pretrain 資料，這點在規劃文件裡已經記錄原因）。
#
# gradient_checkpointing 刻意關掉：smoke test 時發現開著會讓 LoRA/projector 完全學不到東西
# （loss 跟 grad_norm 都卡在精確的 0.0，"None of the inputs have requires_grad=True" 警告
# 出現兩次）——`enable_input_require_grads()` 雖然在 lora_enable 分支前有呼叫，但沒有讓梯度
# 正確傳回 LoRA 參數。手動 forward/backward 測試過，關掉 gradient checkpointing 後 loss/grad
# 都正常。46GB GPU 記憶體對 3B 模型 + LoRA 綽綽有餘，不需要省記憶體，所以直接關掉這個選項而
# 不是深入除錯，比較划算。
#
# model_max_length 設 1024 不是 512：CLIP-ViT-L/14-336 一張圖會展開成 576 個 patch token，
# 加上文字 prompt/回答，序列長度輕鬆超過 512，512 會把 assistant 的回答截斷掉、labels 全部
# 變 -100，導致 loss 變成 NaN（smoke test 時實測到，且 Trainer 的 log 會把 NaN 顯示成誤導性的
# 0.0，要另外印真正的 loss 才看得出來）。
#
# 用法：bash train_lora.sh <LLAMA_MODEL_PATH_OR_HF_ID> <DATA_DIR> <OUTPUT_DIR>
# 例：bash train_lora.sh meta-llama/Llama-3.2-3B /datadrive/VLM/DINO_LLM/X-ray/llava_data \
#       /datadrive/VLM/DINO_LLM/X-ray/stage_llava_ft/checkpoints
set -e

MODEL_PATH=${1:?need base LLM path or HF repo id, e.g. meta-llama/Llama-3.2-3B}
DATA_DIR=${2:?need dir containing train.json/validation.json from build_llava_data.py}
OUTPUT_DIR=${3:?need output checkpoint dir}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 用明確路徑指到 dino_llm conda env 的 python，不能用裸的 python3——這個 shell 預設的
# python3 是 workenv base env，沒裝 transformers/peft 這些套件，跑起來會在
# `from .model import LlavaLlamaForCausalLM` 這行直接 ImportError。
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/workenv/envs/dino_llm/bin/python3}"

"$PYTHON_BIN" -m llava.train.train \
    --model_name_or_path "$MODEL_PATH" \
    --version llama3 \
    --data_path "$DATA_DIR/train.json" \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
