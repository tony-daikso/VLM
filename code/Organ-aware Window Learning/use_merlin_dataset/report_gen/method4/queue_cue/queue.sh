#!/bin/bash
# Method 4 cue-source queue (A6000, user-approved 2026-11-09 "照建議"):
#   wait for Stage 1 (lingshu7b_s1) -> output check -> preflight (2 optimizer steps + 4-case infer per group)
#   -> U_oof -> U_gt -> T_oof -> T_gt  (Stage 2, 2 epochs each, ONE at a time, A6000 only)
#   each group: train (NaN / stall watchdog) -> val 600-subset inference (normal + error-cue 0.5/0.05 seed 20261109)
#   a failed group is recorded and the queue goes on with the next one.
# Qwen re-extraction + metrics run in eval_worker.sh (separate, one evaluation at a time, client of Spark :8003).
# Never touches test, the CG vLLM (:8000) or Spark std14-qwen (:8005).  pin_memory=False is hard-coded in train/infer.
# usage: setsid nohup bash queue.sh > logs/queue.log 2>&1 &     stop: bash stop.sh     progress: bash status.sh
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
CODE=$M4/code_v2
LLM=$TONY/models/Lingshu-7B
FEAT=$M4/features
DATA=$M4/data
SUB=$M4/data_sub/val_sub600.jsonl
CUE=$M4/cue/cue_gt.json
S1=$M4/runs/lingshu7b_s1
S1LOG=$M4/logs/lingshu7b_s1.log
PY=$TONY/tools/vllm-env/bin/python
GROUPS_ORDER="U_oof U_gt T_oof T_gt"
FLIP="--cue-flip-pos2neg 0.5 --cue-flip-neg2pos 0.05 --cue-flip-seed 20261109"
INFER_BATCH=${INFER_BATCH:-32}
MAX_NEW=${MAX_NEW:-512}
STALL_MIN=${STALL_MIN:-60}          # no new log line for this many minutes while training -> kill, mark failed

export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$Q/logs" "$Q/state" "$Q/preflight" "$M4/gen_cue"
echo $$ > "$Q/state/queue.pid"
cd "$CODE"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$Q/STATUS.log"; }
mark() { echo "$(date '+%m-%d %H:%M:%S') ${3:-}" > "$Q/state/$1.$2"; }   # mark <group> <what> [note]
stopped() { [ -e "$Q/STOP" ]; }

common_args() {   # $1 group -> train.py args (no --out)
    local a="--stage 2 --backbone lingshu7b --llm $LLM --data $DATA --meta $FEAT/token_meta.csv --epochs 2"
    a="$a --micro-bs 4 --grad-accum 4 --max-len 1024 --num-workers 2 --val-n 400"
    case $1 in
        U_oof) a="$a --init-adapter $S1/best/adapter.pt --cue-source oof" ;;
        U_gt)  a="$a --init-adapter $S1/best/adapter.pt --cue-source gt_dropout --cue-gt $CUE --cue-gt-map sample --cue-dropout 0.3 --meas-dropout 0" ;;
        T_oof) a="$a --text-only --cue-source oof" ;;
        T_gt)  a="$a --text-only --cue-source gt_dropout --cue-gt $CUE --cue-gt-map sample --cue-dropout 0.3 --meas-dropout 0" ;;
    esac
    echo "$a"
}
textflag() { case $1 in T_*) echo "--text-only" ;; *) echo "" ;; esac; }

wait_gpu() {   # free GPU memory >= $1 GB (the CG vLLM could come back on A6000; wait instead of failing)
    local n=0
    while true; do
        local f
        f=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%d", ($2-$1)/1024}')
        [ "${f:-0}" -ge "$1" ] && return 0
        [ $((n % 10)) -eq 0 ] && log "waiting for GPU: ${f} GB free < $1 GB"
        n=$((n + 1)); stopped && return 1; sleep 60
    done
}

# run a command in its own process group with a NaN / stall watchdog on its log. $1 log, rest = command
run_watched() {
    local lg=$1; shift
    setsid "$@" > "$lg" 2>&1 < /dev/null &
    local pid=$! last mt now
    echo "$pid" > "$Q/state/current.pgid"
    last=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if grep -qiE "loss (nan|inf|-inf)\b|val loss (nan|inf)" "$lg"; then
            log "  NaN/inf loss in $lg -> killing"; kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$Q/state/current.pgid"; return 90
        fi
        mt=$(stat -c %Y "$lg"); now=$(date +%s)
        if [ $((now - mt)) -gt $((STALL_MIN * 60)) ]; then
            log "  no log progress for ${STALL_MIN} min in $lg -> killing"; kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$Q/state/current.pgid"; return 91
        fi
    done
    wait "$pid"; local rc=$?
    rm -f "$Q/state/current.pgid"
    return $rc
}

finite_losses() {   # every "loss X" printed in the log is a finite number and at least one exists
    grep -oE "(val )?loss [^ ]+" "$1" | awk '{v=$NF; if (v !~ /^[0-9.]+(e[-+]?[0-9]+)?$/) bad=1; n++} END {exit (n>0 && !bad) ? 0 : 1}'
}

n_lines() { [ -s "$1" ] && wc -l < "$1" || echo 0; }

# ------------------------------------------------------------------ 0. wait for Stage 1
log "queue started (pid $$); groups: $GROUPS_ORDER; code $CODE"
while pgrep -f "[t]rain.py --stage 1" > /dev/null; do
    stopped && { log "STOP file -> exit while waiting for Stage 1"; exit 0; }
    sleep 120
done
log "Stage 1 process gone"
S1OK=1
for f in "$S1/best/adapter.pt" "$S1/final/adapter.pt" "$S1/final/manifest.json"; do
    [ -s "$f" ] || { log "Stage 1 output missing: $f"; S1OK=0; }
done
grep -q "^done " "$S1LOG" || { log "Stage 1 log has no 'done' line"; S1OK=0; }
finite_losses "$S1LOG" || { log "Stage 1 log has non-finite loss"; S1OK=0; }
grep -q "val loss" "$S1LOG" || { log "Stage 1 log has no val loss"; S1OK=0; }
if [ $S1OK = 1 ]; then
    log "Stage 1 check passed: $(grep 'val loss' "$S1LOG" | tail -1); adapter md5 $(md5sum < "$S1/best/adapter.pt" | cut -c1-12)"
    mark stage1 ok
else
    log "Stage 1 check FAILED -> U groups will be skipped, T groups still run"
    mark stage1 failed
fi

# ------------------------------------------------------------------ 1. preflight (2 steps + 4-case infer per group)
for g in $GROUPS_ORDER; do
    stopped && { log "STOP file -> exit"; exit 0; }
    [ -e "$Q/state/$g.done" ] || [ -e "$Q/state/$g.preflight_ok" ] && continue
    if [[ $g == U_* ]] && [ $S1OK != 1 ]; then mark "$g" failed "no Stage 1 adapter"; continue; fi
    P=$Q/preflight/$g
    rm -rf "$P"; mkdir -p "$P"
    wait_gpu 32 || exit 0
    log "preflight $g: 2 optimizer steps"
    # shellcheck disable=SC2046
    run_watched "$P/train.log" $PY train.py $(common_args "$g") --out "$P/run" --max-steps 2 --train-limit 64 --val-n 8 --log-every 1
    rc=$?
    if [ $rc != 0 ] || ! finite_losses "$P/train.log" || [ ! -s "$P/run/final/manifest.json" ]; then
        log "preflight $g FAILED (rc $rc) -> group skipped; see $P/train.log"; mark "$g" failed "preflight train rc $rc"; continue
    fi
    log "preflight $g: train ok ($(grep -oE '^step [0-9]+/[0-9]+ ep [0-9]+ loss [0-9.]+' "$P/train.log" | tr '\n' ';'))"
    # shellcheck disable=SC2046
    run_watched "$P/infer.log" $PY infer.py --llm "$LLM" --run "$P/run/final" --data "$SUB" --out "$P/gen4.jsonl" \
        --limit 4 --batch 4 --max-new-tokens 64 --cue-gt "$CUE" $FLIP $(textflag "$g")
    rc=$?
    if [ $rc != 0 ] || [ "$(n_lines "$P/gen4.jsonl")" != 4 ] || [ ! -s "$P/gen4.cues.jsonl" ]; then
        log "preflight $g infer FAILED (rc $rc) -> group skipped; see $P/infer.log"; mark "$g" failed "preflight infer rc $rc"; continue
    fi
    log "preflight $g: infer ok"
    find "$P/run" \( -name "adapter.pt" -o -name "*.safetensors" \) -delete     # keep logs/manifests only
    mark "$g" preflight_ok
done

# ------------------------------------------------------------------ 2. the four groups, one at a time
for g in $GROUPS_ORDER; do
    stopped && { log "STOP file -> exit"; exit 0; }
    [ -e "$Q/state/$g.done" ] && { log "$g already done, skip"; continue; }
    [ -e "$Q/state/$g.failed" ] && { log "$g marked failed ($(cat "$Q/state/$g.failed")), skip"; continue; }
    RUN=$M4/runs/$g
    G=$M4/gen_cue/$g
    mkdir -p "$G"
    if [ -s "$RUN/final/manifest.json" ] && [ -s "$RUN/best/manifest.json" ]; then
        log "$g: finished run found in $RUN, not retraining"
    else
        if [ -e "$RUN" ]; then mv "$RUN" "$RUN.partial_$(date +%m%d_%H%M)"; log "$g: moved partial run aside"; fi
        wait_gpu 32 || exit 0
        log "$g: TRAIN start -> $RUN (log $Q/logs/${g}_train.log)"
        mark "$g" training
        # shellcheck disable=SC2046
        run_watched "$Q/logs/${g}_train.log" $PY train.py $(common_args "$g") --out "$RUN" --log-every 20
        rc=$?
        rm -f "$Q/state/$g.training"
        if [ $rc != 0 ] || ! finite_losses "$Q/logs/${g}_train.log" || [ ! -s "$RUN/final/manifest.json" ] || [ ! -s "$RUN/best/manifest.json" ]; then
            log "$g: TRAIN FAILED (rc $rc; 90=NaN, 91=stall) -> next group"; mark "$g" failed "train rc $rc"; continue
        fi
        cp "$Q/logs/${g}_train.log" "$RUN/train.log"
        log "$g: TRAIN ok; $(grep 'val loss' "$Q/logs/${g}_train.log" | tr '\n' ' ')"
    fi
    mark "$g" train_ok
    INF_OK=1
    for kind in normal flip; do
        stopped && { log "STOP file -> exit"; exit 0; }
        out=$G/val600.jsonl; extra=""
        [ $kind = flip ] && { out=$G/val600_flip.jsonl; extra=$FLIP; }
        if [ "$(n_lines "$out")" = 600 ]; then log "$g: $kind inference exists"; continue; fi
        wait_gpu 24 || exit 0
        log "$g: INFER $kind -> $out"
        # shellcheck disable=SC2046
        run_watched "$Q/logs/${g}_infer_$kind.log" $PY infer.py --llm "$LLM" --run "$RUN/best" --data "$SUB" --out "$out" \
            --batch "$INFER_BATCH" --max-new-tokens "$MAX_NEW" --cue-gt "$CUE" $extra $(textflag "$g")
        rc=$?
        if [ $rc != 0 ] || [ "$(n_lines "$out")" != 600 ]; then
            log "$g: INFER $kind FAILED (rc $rc, $(n_lines "$out") rows)"; INF_OK=0
        else
            log "$g: INFER $kind ok ($(grep -c '"findings": {}' "$out") of 600 with no parsed findings)"
        fi
    done
    if [ $INF_OK = 1 ]; then mark "$g" done; else mark "$g" failed "inference"; fi
done
log "all groups processed: $(cd "$Q/state" && ls U_* T_* 2>/dev/null | tr '\n' ' ')"
mark queue finished
