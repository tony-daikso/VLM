#!/bin/bash
# Method 4: full-val (5,055 cases) NORMAL inference for the 4 finished arms (A6000, 2026-11-10).
#   Same invocation as queue.sh's val600 normal inference (runs/<g>/best, --cue-gt for the cue log, no flips,
#   --text-only for T_*), only --data is the full val split. Cues in the val prompts = method-1 OOF predicted cues
#   for every arm. One inference at a time, order U_gt -> T_gt -> U_oof -> T_oof; a group whose output already
#   has 5,055 rows is skipped; a failed group is recorded and the script goes on with the next one.
#   No evaluation here (eval decided later). Never touches test, the CG vLLM (:8000) or the Spark eval tunnel.
#   pin_memory=False is hard-coded in infer.py.
# usage: setsid nohup bash infer_full.sh > logs/infer_full.log 2>&1 < /dev/null &      stop: touch STOP (or kill the pgid in state/current_full.pgid)
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
CODE=$M4/code_v2
LLM=$TONY/models/Lingshu-7B
VAL=$M4/data/val.jsonl
CUE=$M4/cue/cue_gt.json
PY=$TONY/tools/vllm-env/bin/python
GROUPS_ORDER=${GROUPS_ORDER:-"U_gt T_gt U_oof T_oof"}
INFER_BATCH=${INFER_BATCH:-32}
MAX_NEW=${MAX_NEW:-512}
STALL_MIN=${STALL_MIN:-60}
N_VAL=$(wc -l < "$VAL")

export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$Q/logs" "$Q/state"
echo $$ > "$Q/state/infer_full.pid"
cd "$CODE"

log() { echo "[$(date '+%m-%d %H:%M:%S')] [full] $*" | tee -a "$Q/STATUS.log"; }
mark() { echo "$(date '+%m-%d %H:%M:%S') ${3:-}" > "$Q/state/$1.$2"; }
stopped() { [ -e "$Q/STOP" ]; }
n_lines() { [ -s "$1" ] && wc -l < "$1" || echo 0; }
textflag() { case $1 in T_*) echo "--text-only" ;; *) echo "" ;; esac; }

wait_gpu() {   # free GPU memory >= $1 GB
    local n=0 f
    while true; do
        f=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%d", ($2-$1)/1024}')
        [ "${f:-0}" -ge "$1" ] && return 0
        [ $((n % 10)) -eq 0 ] && log "waiting for GPU: ${f} GB free < $1 GB"
        n=$((n + 1)); stopped && return 1; sleep 60
    done
}

run_watched() {   # $1 log, rest = command; own process group + stall watchdog
    local lg=$1; shift
    setsid "$@" > "$lg" 2>&1 < /dev/null &
    local pid=$! mt now
    echo "$pid" > "$Q/state/current_full.pgid"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        mt=$(stat -c %Y "$lg"); now=$(date +%s)
        if [ $((now - mt)) -gt $((STALL_MIN * 60)) ]; then
            log "  no log progress for ${STALL_MIN} min in $lg -> killing"; kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$Q/state/current_full.pgid"; return 91
        fi
    done
    wait "$pid"; local rc=$?
    rm -f "$Q/state/current_full.pgid"
    return $rc
}

log "full-val normal inference started (pid $$): $GROUPS_ORDER; $N_VAL val cases; batch $INFER_BATCH, max-new $MAX_NEW"
for g in $GROUPS_ORDER; do
    stopped && { log "STOP file -> exit"; exit 0; }
    RUN=$M4/runs/$g/best
    G=$M4/gen_cue/$g
    out=$G/val_full.jsonl
    lg=$Q/logs/infer_full_$g.log
    mkdir -p "$G"
    if [ "$(n_lines "$out")" = "$N_VAL" ]; then
        log "$g: full-val output complete ($N_VAL rows), skip"; mark "$g" infer_full_ok "existing"; continue
    fi
    if [ ! -s "$RUN/manifest.json" ]; then
        log "$g: no checkpoint $RUN -> skip"; mark "$g" infer_full_failed "no checkpoint"; continue
    fi
    rm -f "$Q/state/$g.infer_full_failed"
    wait_gpu 24 || exit 0
    log "$g: INFER full-val normal -> $out (ckpt $RUN, log $lg)"
    mark "$g" infer_full_running
    # shellcheck disable=SC2046
    run_watched "$lg" $PY infer.py --llm "$LLM" --run "$RUN" --data "$VAL" --out "$out" \
        --batch "$INFER_BATCH" --max-new-tokens "$MAX_NEW" --cue-gt "$CUE" $(textflag "$g")
    rc=$?
    rm -f "$Q/state/$g.infer_full_running"
    if [ $rc != 0 ] || [ "$(n_lines "$out")" != "$N_VAL" ]; then
        log "$g: INFER full-val FAILED (rc $rc, $(n_lines "$out") rows; 91=stall) -> next group"
        mark "$g" infer_full_failed "rc $rc rows $(n_lines "$out")"
    else
        log "$g: INFER full-val ok ($(grep -c '"findings": {}' "$out") of $N_VAL with no parsed findings)"
        mark "$g" infer_full_ok
    fi
done
log "full-val inference finished: $(cd "$Q/state" && ls ./*.infer_full_* 2>/dev/null | sed 's|^\./||' | tr '\n' ' ')"
mark infer_full finished
