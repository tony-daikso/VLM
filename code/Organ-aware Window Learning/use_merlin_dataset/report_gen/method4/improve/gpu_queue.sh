#!/bin/bash
# Agent F improvement experiments: sequential A6000 GPU job runner (one job at a time).
# Jobs are lines "name|command" in $F/jobs.txt (appendable while the runner is alive; '#' lines ignored).
# A job runs once: state/<name>.ok or state/<name>.failed. NaN / stall watchdog on the job log.
# usage (A6000): setsid nohup bash gpu_queue.sh >> $F/logs/gpu_queue.log 2>&1 < /dev/null &      stop: touch $F/STOP
# Never touches test (infer.py / train.py refuse it), the CG vLLM, or Spark.  pin_memory=False is hard-coded in code.
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
F=$M4/improve
STALL_MIN=${STALL_MIN:-60}
export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$F/state" "$F/logs"
echo $$ > "$F/state/gpu_queue.pid"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [gpu] $*" | tee -a "$F/STATUS.log"; }

wait_gpu() {
    local n=0 f
    while true; do
        f=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%d", ($2-$1)/1024}')
        [ "${f:-0}" -ge "$1" ] && return 0
        [ $((n % 10)) -eq 0 ] && log "waiting for GPU: ${f} GB free < $1 GB"
        n=$((n + 1)); [ -e "$F/STOP" ] && return 1; sleep 60
    done
}

run_watched() {   # $1 log, rest = command string (bash -c)
    local lg=$1; shift
    setsid bash -c "$*" > "$lg" 2>&1 < /dev/null &
    local pid=$! mt now
    echo "$pid" > "$F/state/current.pgid"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if grep -qiE "loss (nan|inf|-inf)\b|val loss (nan|inf)" "$lg"; then
            log "  NaN/inf loss in $lg -> killing"; kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$F/state/current.pgid"; return 90
        fi
        mt=$(stat -c %Y "$lg"); now=$(date +%s)
        if [ $((now - mt)) -gt $((STALL_MIN * 60)) ]; then
            log "  no log progress for ${STALL_MIN} min in $lg -> killing"; kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$F/state/current.pgid"; return 91
        fi
    done
    wait "$pid"; local rc=$?
    rm -f "$F/state/current.pgid"
    return $rc
}

log "gpu queue started (pid $$)"
while true; do
    [ -e "$F/STOP" ] && { log "STOP file -> exit"; exit 0; }
    job=""
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        name=${line%%|*}
        [ -e "$F/state/$name.ok" ] || [ -e "$F/state/$name.failed" ] && continue
        job=$line; break
    done < "$F/jobs.txt"
    if [ -z "$job" ]; then sleep 120; continue; fi
    name=${job%%|*}; cmd=${job#*|}
    wait_gpu 30 || exit 0
    log "$name: start"
    touch "$F/state/$name.running"
    run_watched "$F/logs/$name.log" "$cmd"
    rc=$?
    rm -f "$F/state/$name.running"
    if [ $rc = 0 ]; then
        log "$name: ok ($(grep -E 'val loss' "$F/logs/$name.log" | tr '\n' ' ' | cut -c1-200))"; touch "$F/state/$name.ok"
    else
        log "$name: FAILED rc $rc (90=NaN, 91=stall), see $F/logs/$name.log"; echo "rc $rc" > "$F/state/$name.failed"
    fi
done
