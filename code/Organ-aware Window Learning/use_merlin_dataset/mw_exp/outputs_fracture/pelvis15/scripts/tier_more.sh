#!/bin/bash
# pelvis15 quick tier follow-ups (agent fracture_rp, 11-07 23:05; A6000 GPU): stage 0-3 + maxinst 400, maxinst 600/800, side 1.0.
# tier rows only (train), 5-fold OOF on t_pel, 3 seeds. stop: touch outputs_fracture/pelvis15/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; PY=~/miniconda3/envs/radar/bin/python
run(){ n=$1; shift; [ -f runs/$n/summary.json ] && return; [ -f STOP ] && return
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_probe.py labels.csv "$@" > logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/tier_summary.txt; }
S="--ids ids_tier.txt --seeds 3 --side-csv pel_side.csv"
( run t295s3_m400 f_295s3 runs/t295s3_m400 $S --side 0.5 --maxinst 400; run t295s_m800 f_295 runs/t295s_m800 $S --side 0.5 --maxinst 800 ) &
( run t295s_m600 f_295 runs/t295s_m600 $S --side 0.5 --maxinst 600; run t295s1_m400 f_295 runs/t295s1_m400 $S --side 1.0 --maxinst 400 ) &
wait; echo TIER_MORE_DONE >> logs/tier_summary.txt
