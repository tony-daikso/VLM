#!/bin/bash
# One evaluation (A6000, CPU, nice 19): bash eval_one.sh <U_oof|U_gt|T_oof|T_gt> <normal|flip>
# gen  : m4/gen_cue/<g>/val600[_flip].jsonl (+ .cues.jsonl from infer.py --cue-gt)
# out  : m4/eval/results_cue/<g>_val600[_flip]/{metrics.json, report.md, per_case_categories.jsonl}; log logs/eval_<g>_<kind>.log
# Qwen : Spark merlin-qwen-fp8 via tunnel A6000:18013 -> Spark 127.0.0.1:8003 (client only); cache m4/eval/qwen_extract_cache.jsonl
set -uo pipefail
g=$1; kind=$2
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
PY=$TONY/tools/vllm-env/bin/python
PORT=${EVAL_PORT:-18013}
JOBS=${EVAL_JOBS:-8}
log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval] $*" | tee -a "$Q/STATUS.log"; }
out=$M4/gen_cue/$g/val600.jsonl; name=${g}_val600
[ "$kind" = flip ] && { out=$M4/gen_cue/$g/val600_flip.jsonl; name=${g}_val600_flip; }
RES=$M4/eval/results_cue/$name
[ -s "$RES/metrics.json" ] && { log "$name already evaluated"; exit 0; }
n=$( [ -s "$out" ] && wc -l < "$out" || echo 0 )
[ "$n" = 600 ] || { log "$name: generation not complete ($n rows) -> skip"; exit 0; }

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L $PORT:127.0.0.1:8003 daikso_spark@192.168.0.95 &
TUN=$!
trap 'kill $TUN 2>/dev/null || true' EXIT
ok=0
for i in $(seq 1 12); do      # up to 1 h for the endpoint
    sleep 5
    if $PY -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10)" 2>/dev/null; then ok=1; break; fi
    kill -0 $TUN 2>/dev/null || { ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L $PORT:127.0.0.1:8003 daikso_spark@192.168.0.95 & TUN=$!; }
    sleep 295
done
if [ $ok != 1 ]; then log "$name: Spark Qwen :8003 not reachable for 1 h -> FAILED"; echo "no endpoint" > "$Q/state/$g.eval_${kind}_failed"; exit 1; fi
log "$name: evaluate start (jobs $JOBS)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $M4/code_v2/eval/evaluate.py --gen "$out" --name "$name" \
    --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
    --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl \
    --ids $M4/data_sub/val_qwen_subset_ids.txt --split val \
    --url http://127.0.0.1:$PORT/v1 --jobs "$JOBS" --cache $M4/eval/qwen_extract_cache.jsonl --out-root $M4/eval/results_cue \
    --bertscore-model $TONY/models/roberta-large --bertscore-threads 4 --cue-log "${out%.jsonl}.cues.jsonl" \
    > "$Q/logs/eval_${g}_${kind}.log" 2>&1 < /dev/null
rc=$?
if [ $rc = 0 ] && [ -s "$RES/metrics.json" ]; then
    log "$name: evaluate ok -> $RES/report.md"; rm -f "$Q/state/$g.eval_${kind}_failed"; touch "$Q/state/$g.eval_${kind}_ok"
else
    log "$name: evaluate FAILED (rc $rc), see $Q/logs/eval_${g}_${kind}.log"; echo "rc $rc" > "$Q/state/$g.eval_${kind}_failed"
fi
