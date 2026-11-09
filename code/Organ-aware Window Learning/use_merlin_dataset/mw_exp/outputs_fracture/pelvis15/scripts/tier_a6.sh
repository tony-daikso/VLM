#!/bin/bash
# pelvis15 tier comparison (A6000 GPU): encoders 292/294/295 x {base, side 0.5}, 3 seeds, tier rows (train pelvis+ 309 +
# 1,500 negatives), 5-fold OOF on t_pel. stop: touch outputs_fracture/pelvis15/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; PY=~/miniconda3/envs/radar/bin/python; mkdir -p runs logs
run(){ n=$1; shift; [ -f runs/$n/summary.json ] && return; [ -f STOP ] && return
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_probe.py labels.csv "$@" > logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/tier_summary.txt; }
( run t292 f_292 runs/t292 --ids ids_tier.txt --seeds 3; run t292s f_292 runs/t292s --ids ids_tier.txt --seeds 3 --side 0.5 --side-csv pel_side.csv ) &
( run t294 f_294 runs/t294 --ids ids_tier.txt --seeds 3; run t294s f_294 runs/t294s --ids ids_tier.txt --seeds 3 --side 0.5 --side-csv pel_side.csv ) &
( run t295 f_295 runs/t295 --ids ids_tier.txt --seeds 3; run t295s f_295 runs/t295s --ids ids_tier.txt --seeds 3 --side 0.5 --side-csv pel_side.csv ) &
wait; echo TIER_DONE >> logs/tier_summary.txt
