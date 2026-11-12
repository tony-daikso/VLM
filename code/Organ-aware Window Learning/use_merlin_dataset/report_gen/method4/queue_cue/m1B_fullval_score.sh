#!/bin/bash
# Method-1 1B full-val scoring on the CT-rate box CPU (2026-11-12), from the Qwen cache filled on Spark by
# m1B_fullval_extract.sh (copied to $W/eval/qwen_extract_cache.jsonl). Same evaluate.py call as the peer's
# eval_full2.sh score_arm (--no-request, BERTScore roberta-large layer 17, 4 threads), no cue log (template has no cues).
# Also re-scores the 600-case subset from the same cache (no BERTScore) for the 0.665 check.
# usage (CT-rate): setsid nohup bash m1B_fullval_score.sh > logs/score.out 2>&1 < /dev/null &
set -uo pipefail
W=${W:-/media/daikso/Tony/VLM/report_gen_m1_fullval}
PY=${PY:-/media/daikso/Tony/envs/pillar/bin/python}
BERT=${BERT:-/media/daikso/Tony/models/roberta-large}
GEN=$W/inputs/val_drafts_B_text.jsonl
log() { echo "[$(date '+%m-%d %H:%M:%S')] [m1B-score] $*" | tee -a "$W/logs/STATUS.log"; }
ev() {
    local n=$1; shift
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $W/code/evaluate.py --gen "$GEN" --name "$n" \
        --labels $W/inputs/labels_disease14_all.jsonl --meta $W/inputs/metadata.csv --disease14-dir $W/disease14 \
        --targets $W/inputs/targets_val.jsonl --split val --url http://127.0.0.1:1/v1 \
        --cache $W/eval/qwen_extract_cache.jsonl --out-root $W/eval/results --no-request "$@" < /dev/null
}
log "scoring started (pid $$)"
ev m1B_sub600_recheck --ids $W/inputs/val_qwen_subset_ids.txt >> "$W/logs/score_sub600.log" 2>&1
log "m1B_sub600_recheck rc $?"
ev m1B_valfull --bertscore-model "$BERT" --bertscore-threads 4 >> "$W/logs/score_valfull.log" 2>&1
log "m1B_valfull rc $? -> $W/eval/results/m1B_valfull/report.md"
