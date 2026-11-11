#!/bin/bash
# Full-val (5,055) evaluation of the method-4 normal inference (A6000, CPU, nice 19), sequential:
#   U_gt -> T_gt -> U_oof -> T_oof   (override: bash eval_full.sh T_gt U_oof ...)
# gen  : m4/gen_cue/<g>/val_full.jsonl (+ val_full.cues.jsonl from infer.py --cue-gt)
# out  : m4/eval/results_cue/<g>_valfull/{metrics.json, report.md, per_case_categories.jsonl}; log logs/evalfull_<g>.log
# state: queue_cue/state/<g>.eval_full_ok | <g>.eval_full_failed ; STATUS.log lines "[evalfull]"
# Qwen : Spark merlin-qwen-fp8 via tunnel A6000:18013 -> Spark 127.0.0.1:8003 (client only, jobs 40 by default;
#        11-11: container rebuilt with --max-num-seqs 64 -> 24 slots left for other evaluations);
#        shared cache m4/eval/qwen_extract_cache.jsonl. A keeper loop re-opens the tunnel if ssh dies.
#        If an arm ends with failed extractions, one more pass retries only those (ok rows come from the cache).
# Stop : touch queue_cue/STOP (checked between arms; the running evaluate.py finishes its arm).
# Launch (detached): setsid nohup bash queue_cue/eval_full.sh > queue_cue/logs/eval_full.out 2>&1 < /dev/null &
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
PY=$TONY/tools/vllm-env/bin/python
PORT=${EVAL_PORT:-18013}
JOBS=${EVAL_JOBS:-40}
N_VAL=5055
GROUPS_=${*:-"U_gt T_gt U_oof T_oof"}
log() { echo "[$(date '+%m-%d %H:%M:%S')] [evalfull] $*" | tee -a "$Q/STATUS.log"; }

exec 9>"$Q/state/eval_full.lock"
flock -n 9 || { log "another eval_full.sh holds state/eval_full.lock -> exit"; exit 1; }
echo $$ > "$Q/state/eval_full.pid"
log "eval_full started (pid $$): $GROUPS_; jobs $JOBS"

# wait for any val600 evaluation still using the endpoint
while pgrep -f "[e]val_one.sh" >/dev/null; do sleep 60; done

# ---- tunnel keeper (only the Merlin :8003 service on Spark)
(
    while true; do
        ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
            -L $PORT:127.0.0.1:8003 daikso_spark@192.168.0.95 < /dev/null
        sleep 15
    done
) &
KEEPER=$!
cleanup() { local c; c=$(pgrep -P $KEEPER); kill $KEEPER 2>/dev/null; [ -n "$c" ] && kill $c 2>/dev/null; true; }
trap cleanup EXIT

endpoint_ok() { $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10)" 2>/dev/null; }
wait_endpoint() {        # up to 1 h
    for i in $(seq 1 120); do endpoint_ok && return 0; sleep 30; done
    return 1
}

for g in $GROUPS_; do
    [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
    name=${g}_valfull
    RES=$M4/eval/results_cue/$name
    gen=$M4/gen_cue/$g/val_full.jsonl
    [ -e "$Q/state/$g.eval_full_ok" ] && { log "$name already evaluated -> skip"; continue; }
    [ -e "$Q/state/$g.infer_full_ok" ] || { log "$name: no $g.infer_full_ok -> skip"; continue; }
    n=$( [ -s "$gen" ] && wc -l < "$gen" || echo 0 )
    [ "$n" = $N_VAL ] || { log "$name: generation not complete ($n rows) -> skip"; continue; }
    if ! wait_endpoint; then
        log "$name: Spark Qwen :8003 not reachable for 1 h -> FAILED"; echo "no endpoint" > "$Q/state/$g.eval_full_failed"; continue
    fi
    rm -f "$Q/state/$g.eval_full_failed"
    for pass in 1 2; do
        log "$name: evaluate start pass $pass ($n cases, jobs $JOBS)"
        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $M4/code_v2/eval/evaluate.py --gen "$gen" --name "$name" \
            --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
            --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl \
            --split val \
            --url http://127.0.0.1:$PORT/v1 --jobs "$JOBS" --cache $M4/eval/qwen_extract_cache.jsonl --out-root $M4/eval/results_cue \
            --bertscore-model $TONY/models/roberta-large --bertscore-threads 4 --cue-log "${gen%.jsonl}.cues.jsonl" \
            >> "$Q/logs/evalfull_${g}.log" 2>&1 < /dev/null
        rc=$?
        nf=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['qwen']['n_failed_or_unlabelled'])" "$RES/metrics.json" 2>/dev/null || echo NA)
        log "$name: pass $pass rc $rc, failed/unlabelled extractions $nf"
        [ $rc = 0 ] && [ "$nf" = 0 ] && break
        [ -e "$Q/STOP" ] && break
        [ $rc = 0 ] || wait_endpoint || break
    done
    if [ $rc = 0 ] && [ -s "$RES/metrics.json" ]; then
        log "$name: evaluate ok -> $RES/report.md"; touch "$Q/state/$g.eval_full_ok"
    else
        log "$name: evaluate FAILED (rc $rc), see $Q/logs/evalfull_${g}.log"; echo "rc $rc" > "$Q/state/$g.eval_full_failed"
    fi
done
log "eval_full finished"; touch "$Q/state/eval_full.finished"
