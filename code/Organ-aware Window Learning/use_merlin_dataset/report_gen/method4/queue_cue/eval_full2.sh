#!/bin/bash
# Full-val (5,055) evaluation of the method-4 normal inference, PIPELINED version of eval_full.sh (11-11):
#   Spark Qwen extraction of arm k+1 runs while arm k is scored on the A6000 CPU (BERTScore is ~1 h per arm there),
#   so the Spark endpoint never waits for the CPU.
#   per arm: (1) evaluate.py --extract-only (up to 3 passes; later passes only retry failed extractions)
#            (2) background: evaluate.py --no-request (all metrics + BERTScore) -> <g>.eval_full_ok
#                then fill14_eval.sh valfull <g> (improvement 1 "fill 14", merge scoring; validated on val600)
#   at most MAX_SCORERS (2) background scorers at a time (A6000 RAM 53 GB, CPU shared with training).
# gen  : m4/gen_cue/<g>/val_full.jsonl (+ val_full.cues.jsonl)
# out  : m4/eval/results_cue/<g>_valfull/ ; fill14: m4/eval/results_fill14/<g>_valfull_fill_merge/
# state: queue_cue/state/<g>.eval_full_ok | .eval_full_failed ; STATUS.log lines "[evalfull2]"
# Qwen : tunnel A6000:18013 -> Spark merlin-qwen-fp8 (max-num-seqs 64), jobs EVAL_JOBS (40); cache shared.
# Stop : touch queue_cue/STOP (checked between arms).
# Launch: setsid nohup bash queue_cue/eval_full2.sh > queue_cue/logs/eval_full2.out 2>&1 < /dev/null &
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
PY=$TONY/tools/vllm-env/bin/python
PORT=${EVAL_PORT:-18013}
JOBS=${EVAL_JOBS:-40}
MAX_SCORERS=${MAX_SCORERS:-2}
N_VAL=5055
GROUPS_=${*:-"U_gt T_gt U_oof T_oof"}
log() { echo "[$(date '+%m-%d %H:%M:%S')] [evalfull2] $*" | tee -a "$Q/STATUS.log"; }

exec 9>"$Q/state/eval_full.lock"
flock -n 9 || { log "another eval_full holds state/eval_full.lock -> exit"; exit 1; }
echo $$ > "$Q/state/eval_full.pid"
log "eval_full2 started (pid $$): $GROUPS_; jobs $JOBS, scorers $MAX_SCORERS"

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
wait_endpoint() { for i in $(seq 1 120); do endpoint_ok && return 0; sleep 30; done; return 1; }

ev() {  # g extra-args...
    local g=$1; shift
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $M4/code_v2/eval/evaluate.py --gen "$M4/gen_cue/$g/val_full.jsonl" \
        --name "${g}_valfull" --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl \
        --meta $TONY/VLM/merlin_disease14/metadata.csv --disease14-dir $TONY/VLM/merlin_disease14 \
        --targets $M4/inputs/targets_v2/targets_val.jsonl --split val --url http://127.0.0.1:$PORT/v1 --jobs "$JOBS" \
        --cache $M4/eval/qwen_extract_cache.jsonl --out-root $M4/eval/results_cue "$@" < /dev/null
}

score_arm() {  # background: full scoring from the cache, then fill14 (merge) on the same arm
    local g=$1 RES=$M4/eval/results_cue/${1}_valfull
    ev "$g" --no-request --bertscore-model $TONY/models/roberta-large --bertscore-threads 4 \
        --cue-log "$M4/gen_cue/$g/val_full.cues.jsonl" >> "$Q/logs/evalfull_${g}.log" 2>&1
    local rc=$?
    local nf
    nf=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['qwen']['n_failed_or_unlabelled'])" "$RES/metrics.json" 2>/dev/null || echo NA)
    if [ $rc = 0 ] && [ -s "$RES/metrics.json" ]; then
        log "${g}_valfull: scored (failed/unlabelled extractions $nf) -> $RES/report.md"; touch "$Q/state/$g.eval_full_ok"
        bash "$Q/fill14_eval.sh" valfull "$g" >> "$Q/logs/fill14_valfull.out" 2>&1
        log "${g}_valfull: fill14 merge scoring rc $?"
    else
        log "${g}_valfull: scoring FAILED (rc $rc)"; echo "score rc $rc" > "$Q/state/$g.eval_full_failed"
    fi
}

SCORERS=()
for g in $GROUPS_; do
    [ -e "$Q/STOP" ] && { log "STOP file -> no new arms"; break; }
    [ -e "$Q/state/$g.eval_full_ok" ] && { log "${g}_valfull already evaluated -> skip"; continue; }
    [ -e "$Q/state/$g.infer_full_ok" ] || { log "${g}_valfull: no $g.infer_full_ok -> skip"; continue; }
    n=$(wc -l < "$M4/gen_cue/$g/val_full.jsonl" 2>/dev/null || echo 0)
    [ "$n" = $N_VAL ] || { log "${g}_valfull: generation not complete ($n rows) -> skip"; continue; }
    rm -f "$Q/state/$g.eval_full_failed"
    for pass in 1 2 3; do
        wait_endpoint || { log "${g}_valfull: endpoint down for 1 h"; break; }
        log "${g}_valfull: extract pass $pass (jobs $JOBS)"
        ev "$g" --extract-only >> "$Q/logs/evalfull_${g}.log" 2>&1
        nf=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['n_failed_or_unlabelled'])" \
             "$M4/eval/results_cue/${g}_valfull/extract_status.json" 2>/dev/null || echo NA)
        log "${g}_valfull: extract pass $pass done, failed/unlabelled $nf"
        [ "$nf" = 0 ] && break
    done
    while :; do   # free scorer slot
        alive=(); for p in "${SCORERS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
        SCORERS=("${alive[@]}")
        [ ${#SCORERS[@]} -lt "$MAX_SCORERS" ] && break
        sleep 60
    done
    score_arm "$g" &
    SCORERS+=($!)
    log "${g}_valfull: scoring started in background (pid $!)"
done
for p in "${SCORERS[@]}"; do wait "$p"; done
log "eval_full2 finished"; touch "$Q/state/eval_full.finished"
