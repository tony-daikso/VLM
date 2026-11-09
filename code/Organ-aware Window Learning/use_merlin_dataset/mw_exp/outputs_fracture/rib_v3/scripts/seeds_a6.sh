#!/bin/bash
# rib_v3 production seeds (A6000): side 0.5 + lev 0.3 (best train OOF), 5 seeds per run; lane arg = seed0 list
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_v3; PY=~/miniconda3/envs/radar/bin/python
for s in "$@"; do n=v3_sl_s$s; [ -f runs/$n/summary.json ] && continue; [ -f STOP ] && break
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 5 --seed0 $s --side 0.5 --lev 0.3 >> logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/abl_summary.txt; done
