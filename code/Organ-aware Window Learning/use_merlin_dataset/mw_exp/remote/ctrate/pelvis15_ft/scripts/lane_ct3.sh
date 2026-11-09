#!/bin/bash
# 10:20 rebalance (agent pelvis15_ft): A6000 GPU is shared with another agent (rs_feat.py) and runs ~2x slower, (10:25: job 3:1 goes to the idle Spark instead; the 3080 is also shared with rs_ft.py)
# from the A6000 to this 3080. Waits for the orphaned 1:1 training process to end, then 1:1 (skipped if done), 2:1, 3:1.
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft
while pgrep -f "[p]v15_ft.py" > /dev/null; do sleep 60; done
bash scripts/full_run.sh ct 297s "--side 0.5" 1:1 2:1
