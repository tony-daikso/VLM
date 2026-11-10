#!/bin/bash
# Evaluation worker for queue.sh (A6000, CPU only, nice 19): for U_oof -> U_gt -> T_oof -> T_gt, once a group's
# 600-case val inference is finished, run evaluate.py on the normal and the error-cue drafts, ONE evaluation at a time.
# Qwen re-extraction = client of the existing Spark merlin-qwen-fp8 vLLM (127.0.0.1:8003 on Spark) through an ssh
# tunnel A6000:18013 -> Spark:8003 (opened per evaluation, closed afterwards). Starts / stops / changes no server.
# A failed evaluation is recorded (state/<g>.eval_<kind>_failed) and can be redone later: bash eval_one.sh <g> <kind>
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
GROUPS_ORDER="U_oof U_gt T_oof T_gt"
mkdir -p "$Q/logs" "$Q/state"
echo $$ > "$Q/state/eval_worker.pid"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval] $*" | tee -a "$Q/STATUS.log"; }
queue_alive() { [ -s "$Q/state/queue.pid" ] && kill -0 "$(cat "$Q/state/queue.pid")" 2>/dev/null; }

log "eval worker started (pid $$)"
for g in $GROUPS_ORDER; do
    while [ ! -e "$Q/state/$g.done" ] && [ ! -e "$Q/state/$g.failed" ]; do
        [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
        if ! queue_alive && [ ! -e "$Q/state/queue.finished" ]; then
            log "queue.sh is not running and not finished -> evaluating whatever exists for $g"; break
        fi
        sleep 300
    done
    for kind in normal flip; do
        [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
        bash "$Q/eval_one.sh" "$g" "$kind"
    done
done
log "eval worker finished"
touch "$Q/state/eval_worker.finished"
