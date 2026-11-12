#!/bin/bash
# Agent G: proxy evaluation (mt_lib rules, no Qwen, no BERTScore) of one val600 generation, same evaluate.py options as
# eval_queue.sh. CPU only, nice 19, 1 thread (~10 s).
# usage (A6000): bash proxy_run.sh <name> <gen.jsonl> [baseline result names ...]
# PROXY_SUMMARY_FLAGS (optional, default empty = unchanged): extra summarize.py flags, e.g. --hy-first (agent I)
# -> $M4/eval/results_proxy/<name>/{metrics.json,report.md,per_case_categories.jsonl}
#    $M4/improve/proxy/<name>.md : attribute-first summary table, baselines (results_proxy/<b>) + this run
set -euo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
PY=$TONY/tools/vllm-env/bin/python
name=$1; gen=$2; shift 2
cuelog=""
[ -s "${gen%.jsonl}.cues.jsonl" ] && cuelog="--cue-log ${gen%.jsonl}.cues.jsonl"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY $M4/code_f/improve/proxy_eval.py --gen "$gen" --name "$name" \
    --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
    --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl \
    --ids $M4/data_sub/val_qwen_subset_ids.txt --split val --out-root $M4/eval/results_proxy $cuelog
mkdir -p $M4/improve/proxy
cols=""
for b in "$@" "$name"; do [ -s "$M4/eval/results_proxy/$b/metrics.json" ] && cols="$cols $b=$M4/eval/results_proxy/$b/metrics.json"; done
{ echo "# 代理評估（mt_lib 規則，未用 Qwen；只供篩選）：$name  $(date '+%m-%d %H:%M')"; echo
  $PY $M4/code_f/improve/summarize.py $cols --attr-first --per-disease ${PROXY_SUMMARY_FLAGS:-}; } > $M4/improve/proxy/$name.md
echo "[$(date '+%m-%d %H:%M:%S')] [proxy] $name: macro F1 $(python3 -c "import json; m=json.load(open('$M4/eval/results_proxy/$name/metrics.json')); print(round([p for p in m['presence'] if p['definition']=='owl' and p['mode']=='strict' and p['disease']=='ALL_macro'][0]['F1'],3))") -> improve/proxy/$name.md" >> $M4/improve/STATUS.log
