#!/bin/bash
# usage: lane.sh ENC  -> tier runs t$1 (base) and t$1s (side 0.5), 3 seeds, A6000 GPU
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; PY=~/miniconda3/envs/radar/bin/python; e=$1
for v in "" s; do n=t$e$v; [ -f runs/$n/summary.json ] && continue; [ -f STOP ] && break
  X=""; [ "$v" = s ] && X="--side 0.5 --side-csv pel_side.csv"
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_probe.py labels.csv f_$e runs/$n --ids ids_tier.txt --seeds 3 $X > logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/tier_summary.txt; done
