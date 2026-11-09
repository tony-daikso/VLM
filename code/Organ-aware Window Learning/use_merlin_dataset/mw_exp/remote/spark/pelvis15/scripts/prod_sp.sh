#!/bin/bash
# pelvis15 production (agent fracture_rp, 11-07 23:05; Spark GPU): Dataset295 stage0-1 feats, side 0.5 weak sup, maxinst from arg; full train 5-fold OOF + val (fold-model mean).
# usage: prod_sp.sh MAXINST SEED0 [SEED0 ...]   stop: touch ~/Desktop/Tony/OWL/pelvis15/STOP_PROD
cd ~/Desktop/Tony/OWL/pelvis15; PY=~/radar_env/bin/python; m=$1; shift
for s in "$@"; do n=p295s_m${m}_s$s; [ -f runs/$n/summary.json ] && continue; [ -f STOP_PROD ] && break
  OMP_NUM_THREADS=4 nice -n 5 $PY scripts/pv15_probe.py labels.csv f_295 runs/$n --seeds 5 --seed0 $s --side 0.5 --side-csv pel_side.csv --maxinst $m > logs/$n.log 2>&1
  grep PROBE logs/$n.log >> logs/prod_summary.txt; done
