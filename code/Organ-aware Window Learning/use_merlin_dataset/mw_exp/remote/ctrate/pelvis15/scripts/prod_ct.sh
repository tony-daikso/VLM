#!/bin/bash
# pelvis15 production (agent fracture_rp, 11-07 23:10; CT-rate 3080): Dataset295 stage0-1 feats, side 0.5, maxinst arg; full train 5-fold OOF + val.
# usage: prod_ct.sh MAXINST SEED0 [SEED0 ...]   stop: touch /media/daikso/Tony/VLM/data/CT/merlin/pelvis15/STOP_PROD
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15; PY=~/miniconda3/envs/radar/bin/python; m=$1; shift; mkdir -p runs
for s in "$@"; do n=p295s_m${m}_s$s; [ -f runs/$n/summary.json ] && continue; [ -f STOP_PROD ] && break
  OMP_NUM_THREADS=4 nice -n 5 $PY scripts/pv15_probe.py labels.csv f_295 runs/$n --seeds 5 --seed0 $s --side 0.5 --side-csv pel_side.csv --maxinst $m > logs/$n.log 2>&1
  grep PROBE logs/$n.log >> logs/prod_summary.txt; done
