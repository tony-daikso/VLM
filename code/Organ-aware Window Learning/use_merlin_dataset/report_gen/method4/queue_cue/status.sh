#!/bin/bash
# Progress of the method 4 cue-source queue:  ssh A6000 'bash ~/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue/status.sh'
M4=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
alive() { [ -s "$Q/state/$1.pid" ] && kill -0 "$(cat "$Q/state/$1.pid")" 2>/dev/null && echo running || echo "not running"; }
echo "queue.sh: $(alive queue)   eval_worker.sh: $(alive eval_worker)   STOP file: $([ -e "$Q/STOP" ] && echo yes || echo no)"
echo "--- Stage 1: $(grep -E '^step|val loss|^done' $M4/logs/lingshu7b_s1.log | tail -1)"
echo "--- state markers"; (cd "$Q/state" && ls -1 | grep -vE '\.pid$|current.pgid' | sed 's/^/  /')
for g in U_oof U_gt T_oof T_gt; do
    f=$Q/logs/${g}_train.log
    [ -s "$f" ] && echo "--- $g train: $(grep -E '^step|val loss|^done' "$f" | tail -1)"
done
echo "--- last 15 status lines"; tail -15 "$Q/STATUS.log"
echo "--- GPU"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
free -g | head -2
