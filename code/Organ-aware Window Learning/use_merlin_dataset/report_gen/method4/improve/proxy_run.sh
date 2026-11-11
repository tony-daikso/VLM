#!/bin/bash
# Agent G: proxy evaluation (mt_lib rules, no Qwen, no BERTScore) of one val600 generation, same evaluate.py options as
# eval_queue.sh. CPU only, nice 19, 1 thread. usage (A6000): bash proxy_run.sh <name> <gen.jsonl>
# -> $M4/eval/results_proxy/<name>/{metrics.json,report.md,per_case_categories.jsonl}
set -euo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
PY=$TONY/tools/vllm-env/bin/python
name=$1; gen=$2
cuelog=""
[ -s "${gen%.jsonl}.cues.jsonl" ] && cuelog="--cue-log ${gen%.jsonl}.cues.jsonl"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY $M4/code_f/improve/proxy_eval.py --gen "$gen" --name "$name" \
    --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
    --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl \
    --ids $M4/data_sub/val_qwen_subset_ids.txt --split val --out-root $M4/eval/results_proxy $cuelog
