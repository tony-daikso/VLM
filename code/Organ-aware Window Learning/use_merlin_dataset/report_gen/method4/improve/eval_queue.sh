#!/bin/bash
# Agent F improvement experiments: evaluation worker (A6000 CPU, nice 19; Qwen on Spark merlin-qwen-fp8, client only).
# Items are lines "name|gen_jsonl|gpu_job_it_waits_for" in $F/evals.txt (appendable). One evaluation at a time.
# Own tunnel (A6000:18023 -> Spark 127.0.0.1:8003) and own Qwen cache ($M4/eval/qwen_extract_cache_F.jsonl), so it never
# writes agent E's cache. Results: $M4/eval/results_f/<name>/{metrics.json,report.md}. Same evaluate.py / options as
# queue_cue/eval_one.sh (600-case subset ids, owl + B definitions, BERTScore roberta-large).
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
F=$M4/improve
PY=$TONY/tools/vllm-env/bin/python
PORT=${EVAL_PORT:-18023}
JOBS=${EVAL_JOBS:-8}
CACHE=$M4/eval/qwen_extract_cache_F.jsonl
OUT=$M4/eval/results_f
mkdir -p "$F/state" "$F/logs" "$OUT"
echo $$ > "$F/state/eval_queue.pid"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval] $*" | tee -a "$F/STATUS.log"; }

TUN=""
tunnel_up() {
    if [ -n "$TUN" ] && kill -0 "$TUN" 2>/dev/null && \
       $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10)" 2>/dev/null; then return 0; fi
    [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L $PORT:127.0.0.1:8003 daikso_spark@192.168.0.95 &
    TUN=$!
    for i in $(seq 1 12); do
        sleep 5
        $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10)" 2>/dev/null && return 0
        sleep 25
    done
    return 1
}
trap '[ -n "$TUN" ] && kill $TUN 2>/dev/null || true' EXIT

log "eval queue started (pid $$), port $PORT, jobs $JOBS, cache $CACHE"
while true; do
    [ -e "$F/STOP" ] && { log "STOP file -> exit"; exit 0; }
    item=""
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        IFS='|' read -r name gen dep <<< "$line"
        [ -e "$F/state/eval_$name.ok" ] || [ -e "$F/state/eval_$name.failed" ] && continue
        if [ -e "$F/state/$dep.failed" ]; then echo "gpu job $dep failed" > "$F/state/eval_$name.failed"; continue; fi
        [ -e "$F/state/$dep.ok" ] || continue
        item=$line; break
    done < "$F/evals.txt"
    if [ -z "$item" ]; then sleep 120; continue; fi
    IFS='|' read -r name gen dep <<< "$item"
    n=$( [ -s "$gen" ] && wc -l < "$gen" || echo 0 )
    if [ "$n" != 600 ]; then log "$name: $gen has $n rows -> failed"; echo "rows $n" > "$F/state/eval_$name.failed"; continue; fi
    if ! tunnel_up; then log "Spark Qwen :8003 not reachable -> retry in 10 min"; sleep 600; continue; fi
    cuelog=""
    [ -s "${gen%.jsonl}.cues.jsonl" ] && cuelog="--cue-log ${gen%.jsonl}.cues.jsonl"
    log "$name: evaluate start"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $M4/code_f/eval/evaluate.py --gen "$gen" --name "$name" \
        --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
        --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl \
        --ids $M4/data_sub/val_qwen_subset_ids.txt --split val \
        --url http://127.0.0.1:$PORT/v1 --jobs "$JOBS" --cache "$CACHE" --out-root "$OUT" \
        --bertscore-model $TONY/models/roberta-large --bertscore-threads 4 $cuelog \
        > "$F/logs/eval_$name.log" 2>&1 < /dev/null
    rc=$?
    if [ $rc = 0 ] && [ -s "$OUT/$name/metrics.json" ]; then
        log "$name: evaluate ok"; touch "$F/state/eval_$name.ok"
    else
        log "$name: evaluate FAILED rc $rc (see logs/eval_$name.log); retry once later"
        if [ -e "$F/state/eval_$name.retry" ]; then echo "rc $rc" > "$F/state/eval_$name.failed"; else touch "$F/state/eval_$name.retry"; sleep 300; fi
    fi
done
