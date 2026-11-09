#!/bin/bash
# After Stage 1 (A6000): greedy drafts for the first N val cases with the best Stage 1 adapter, then evaluate.py
# (Qwen re-extraction through an A6000 -> Spark tunnel to merlin-qwen-fp8, text metrics, BERTScore on CPU).
# Everything stays on A6000 (drafts + results under m4/eval/).   usage: bash post_stage1.sh [RUN] [N]
set -euo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
RUN=${1:-lingshu7b_s1}
N=${2:-200}
OUT=$M4/eval
mkdir -p "$OUT"
export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib
PY=$TONY/tools/vllm-env/bin/python
cd "$M4/code"
if pgrep -f "[t]rain.py --stage" > /dev/null; then echo "a train.py is still running"; exit 1; fi
$PY infer.py --llm $TONY/models/Lingshu-7B --run "$M4/runs/$RUN/best" --data "$M4/data/val.jsonl" --limit "$N" \
    --out "$OUT/${RUN}_val$N.jsonl" --batch 16 --max-new-tokens 320
# Qwen through Spark (vLLM binds 127.0.0.1:8003 there); tunnel from A6000, closed at the end
ssh -N -o ExitOnForwardFailure=yes -L 18003:127.0.0.1:8003 daikso_spark@192.168.0.95 &
TUN=$!
trap 'kill $TUN 2>/dev/null || true' EXIT
sleep 3
CUDA_VISIBLE_DEVICES= $PY $TONY/VLM/OWL/report_gen/eval/code/evaluate.py --gen "$OUT/${RUN}_val$N.jsonl" --name "${RUN}_val$N" \
    --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
    --disease14-dir $TONY/VLM/merlin_disease14 --targets "$M4/inputs/targets_v2/targets_val.jsonl" \
    --url http://127.0.0.1:18003/v1 --jobs 8 --cache "$OUT/qwen_extract_cache.jsonl" --out-root "$OUT/results" \
    --bertscore-model $TONY/models/roberta-large --bertscore-threads 6
echo "POST_DONE $OUT/results/${RUN}_val$N"
