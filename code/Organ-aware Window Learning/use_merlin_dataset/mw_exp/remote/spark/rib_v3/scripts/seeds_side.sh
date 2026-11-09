#!/bin/bash
cd ~/Desktop/Tony/OWL/rib_v3; PY=~/radar_env/bin/python
for s in 10 20; do n=v3_side_s$s; [ -f runs/$n/summary.json ] && continue; [ -f STOP ] && break
  $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 5 --seed0 $s --side 0.5 > logs/$n.log 2>&1; done
