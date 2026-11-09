#!/bin/bash
cd ~/Desktop/Tony/OWL/rib_v3; PY=~/radar_env/bin/python
while pgrep -f "seeds_side.sh" >/dev/null; do sleep 60; done
n=v3_s1l_s50; [ -f runs/$n/summary.json ] || $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 5 --seed0 50 --side 1.0 --lev 0.3 > logs/$n.log 2>&1
