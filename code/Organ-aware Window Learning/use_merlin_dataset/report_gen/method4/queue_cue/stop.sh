#!/bin/bash
# Stop the method 4 cue-source queue (queue.sh, eval_worker.sh, the running train/infer and evaluate).
# Does NOT touch Stage 1 (lingshu7b_s1), the CG vLLM or any Spark container.
#   ssh A6000 'bash ~/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue/stop.sh'
# Restart later: rm STOP, then start.sh (finished groups are skipped; a half-trained run is moved to runs/<g>.partial_*).
Q=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue
touch "$Q/STOP"
for p in queue eval_worker; do
    [ -s "$Q/state/$p.pid" ] && kill "$(cat "$Q/state/$p.pid")" 2>/dev/null && echo "killed $p"
done
if [ -s "$Q/state/current.pgid" ]; then
    kill -TERM -- -"$(cat "$Q/state/current.pgid")" 2>/dev/null && echo "killed train/infer group $(cat "$Q/state/current.pgid")"
    rm -f "$Q/state/current.pgid"
fi
pkill -f "code_v2/eval/evaluate.py" && echo "killed evaluate.py"
pkill -f "18013:127.0.0.1:8003" && echo "closed tunnel"
rm -f "$Q/state/"*.training
echo "[$(date '+%m-%d %H:%M:%S')] stopped by stop.sh" >> "$Q/STATUS.log"
