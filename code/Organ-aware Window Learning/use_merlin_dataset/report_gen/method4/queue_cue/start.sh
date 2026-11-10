#!/bin/bash
# Start (or resume) the method 4 cue-source queue + its evaluation worker on A6000.
#   ssh A6000 'bash ~/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue/start.sh'
Q=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue
mkdir -p "$Q/logs" "$Q/state"
[ -e "$Q/STOP" ] && { echo "STOP file present ($Q/STOP); remove it first"; exit 1; }
for p in queue eval_worker; do
    if [ -s "$Q/state/$p.pid" ] && kill -0 "$(cat "$Q/state/$p.pid")" 2>/dev/null; then echo "$p already running"; exit 1; fi
done
rm -f "$Q/state/queue.finished" "$Q/state/eval_worker.finished"
setsid nohup bash "$Q/queue.sh" >> "$Q/logs/queue.log" 2>&1 < /dev/null &
sleep 2
setsid nohup bash "$Q/eval_worker.sh" >> "$Q/logs/eval_worker.log" 2>&1 < /dev/null &
sleep 2
echo "queue pid $(cat "$Q/state/queue.pid"), eval worker pid $(cat "$Q/state/eval_worker.pid")"
