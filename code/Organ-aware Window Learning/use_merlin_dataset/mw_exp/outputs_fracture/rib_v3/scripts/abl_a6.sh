#!/bin/bash
# rib_v3 ablation, second wave (A6000 GPU, 200 W cap kept; fold-level resume). stop: touch outputs_fracture/rib_v3/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_v3; PY=~/miniconda3/envs/radar/bin/python
run(){ n=$1; shift; [ -f runs/$n/summary.json ] && return; [ -f STOP ] && return
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 3 "$@" >> logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/abl_summary.txt; }
( run v3_side --side 0.5; run v3_ci_side_ema --ctxin --side 0.5 --ema 0.995 ) &
( run v3_ci_side_lev --ctxin --side 0.5 --lev 0.3; run v3_side_lev --side 0.5 --lev 0.3 ) &
wait; echo ABL_A6_DONE >> logs/abl_summary.txt
