#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; PY=~/miniconda3/envs/radar/bin/python
for m in "$@"; do n=t295s_m$m; [ -f runs/$n/summary.json ] && continue
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_probe.py labels.csv f_295 runs/$n --ids ids_tier.txt --seeds 3 --side 0.5 --side-csv pel_side.csv --maxinst $m > logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/tier_summary.txt; done
