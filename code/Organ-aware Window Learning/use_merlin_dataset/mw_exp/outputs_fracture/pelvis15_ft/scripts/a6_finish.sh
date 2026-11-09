#!/bin/bash
# 10:20 rebalance (agent pelvis15_ft): job 3:1 moved to the CT-rate 3080; waits for the orphaned 2:0 training, then marks the A6000 lane done.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft
while pgrep -f "[p]v15_ft.py --labels.*full_297s_s0" > /dev/null; do sleep 60; done
if [ -f runs/full_297s_s0/fold2.npz ]; then echo "$(date "+%m-%d %H:%M") a6 runs/full_297s_s0 fold 2 done $(grep "fold 2 OOF" logs/full_297s_s0_f2.log | tail -1)" >> logs/full.log; echo "$(date "+%m-%d %H:%M") a6 lane done: 0:0 1:0 2:0 (3:1 moved to ct)" >> logs/full.log
else echo "$(date "+%m-%d %H:%M") a6 job 2:0 FAILED (no fold file after process ended)" >> logs/full.log; fi
