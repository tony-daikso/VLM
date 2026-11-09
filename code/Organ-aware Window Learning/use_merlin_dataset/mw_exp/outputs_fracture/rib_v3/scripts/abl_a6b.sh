#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_v3; PY=~/miniconda3/envs/radar/bin/python
run(){ n=$1; shift; [ -f runs/$n/summary.json ] && return; [ -f STOP ] && return
  OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 3 "$@" >> logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/abl_summary.txt; }
run v3_side1 --side 1.0; run v3_side2 --side 2.0
