#!/bin/bash
# Eval worker v3 (11-10 12:40): never idle while something is evaluable. Priority:
# T_oof/T_gt/U_gt normal first, then U_gt/T_oof/T_gt flip; picks the highest-priority item whose group is done.
# Waits for an already-running eval_one.sh first. One evaluation at a time (Spark Qwen, 8 jobs).
set -uo pipefail
M4=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
echo $$ > "$Q/state/eval_worker.pid"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval] $*" | tee -a "$Q/STATUS.log"; }
ITEMS="U_gt:normal T_oof:normal T_gt:normal U_gt:flip T_oof:flip T_gt:flip"
log "eval worker v3 started (pid $$): run any ready item, normals first"
while pgrep -f "[e]val_one.sh" >/dev/null; do sleep 60; done
while true; do
    [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
    left=0; ran=0
    for item in $ITEMS; do
        g=${item%%:*}; kind=${item##*:}
        [ -e "$Q/state/$g.eval_${kind}_ok" ] && continue
        [ -e "$Q/state/$g.eval_${kind}_failed" ] && continue
        [ -e "$Q/state/$g.failed" ] && continue
        left=1
        if [ -e "$Q/state/$g.done" ]; then bash "$Q/eval_one.sh" "$g" "$kind"; ran=1; break; fi
    done
    [ $left = 0 ] && break
    [ $ran = 0 ] && sleep 300
done
log "eval worker v3 finished"; touch "$Q/state/eval_worker.finished"
