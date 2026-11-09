#!/bin/bash
# Method 4 Stage 1 on A6000 (LLM frozen, only the OWL adapter = type embedding + Perceiver resampler + projector).
# Start ONLY after the user approved it.  usage:  bash start_stage1.sh [RUN_NAME]
# Log: ~/Desktop/Tony/VLM/OWL/report_gen/m4/logs/<RUN_NAME>.log ; weights: m4/runs/<RUN_NAME>/{epoch1,best,final}/
set -euo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
RUN=${1:-lingshu7b_s1}
DATA=$M4/data
FEAT=$M4/features

# ---- pre-flight checks (refuse instead of guessing)
[ -s "$FEAT/manifest.csv" ] || { echo "features not packed ($FEAT/manifest.csv missing)"; exit 1; }
[ -s "$DATA/train.jsonl" ] && [ -s "$DATA/val.jsonl" ] || { echo "train/val jsonl missing in $DATA"; exit 1; }
[ -e "$M4/runs/$RUN" ] && { echo "run dir $M4/runs/$RUN exists; pick another name"; exit 1; }
if pgrep -f "train.py --stage" > /dev/null; then echo "another train.py is running"; exit 1; fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "train rows $(wc -l < "$DATA/train.jsonl"), val rows $(wc -l < "$DATA/val.jsonl")"

export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$M4/logs" "$M4/runs"
cd "$M4/code"
setsid nohup $TONY/tools/vllm-env/bin/python train.py --stage 1 \
    --backbone lingshu7b --llm $TONY/models/Lingshu-7B \
    --data "$DATA" --meta "$FEAT/token_meta.csv" --out "$M4/runs/$RUN" \
    --epochs 1 --micro-bs 4 --grad-accum 4 --lr 1e-3 --max-len 1024 \
    --num-workers 2 --val-n 400 --log-every 20 \
    > "$M4/logs/$RUN.log" 2>&1 < /dev/null &
echo "started pid $! -> $M4/logs/$RUN.log"
