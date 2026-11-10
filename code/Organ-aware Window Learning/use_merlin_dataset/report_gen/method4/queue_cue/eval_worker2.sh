#!/bin/bash
# Re-ordered eval worker (11-10): normal evals first (U_gt, T_oof, T_gt), then flips (U_gt, T_oof, T_gt).
# Waits for any running evaluate.py (from the old worker) to finish first. One evaluation at a time, same eval_one.sh.
set -uo pipefail
M4=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
echo $$ > "$Q/state/eval_worker.pid"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval] $*" | tee -a "$Q/STATUS.log"; }
queue_alive() { [ -s "$Q/state/queue.pid" ] && kill -0 "$(cat "$Q/state/queue.pid")" 2>/dev/null; }
log "eval worker v2 started (pid $$): order U_gt/T_oof/T_gt normal, then flips"
while pgrep -f "code_v2/eval/evaluate.py" >/dev/null; do sleep 120; done
for item in U_gt:normal T_oof:normal T_gt:normal U_gt:flip T_oof:flip T_gt:flip; do
    g=${item%%:*}; kind=${item##*:}
    [ -e "$Q/state/$g.eval_${kind}_ok" ] && continue
    while [ ! -e "$Q/state/$g.done" ] && [ ! -e "$Q/state/$g.failed" ]; do
        [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
        if ! queue_alive && [ ! -e "$Q/state/queue.finished" ]; then log "queue.sh not running -> evaluating whatever exists for $g"; break; fi
        sleep 300
    done
    [ -e "$Q/STOP" ] && { log "STOP file -> exit"; exit 0; }
    bash "$Q/eval_one.sh" "$g" "$kind"
done
log "eval worker v2 finished"; touch "$Q/state/eval_worker.finished"
