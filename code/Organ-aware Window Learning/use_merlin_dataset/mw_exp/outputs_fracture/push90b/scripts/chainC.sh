#!/bin/bash
F=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; B=$F/push90b; cd $B; PY=/home/advantech/miniconda3/envs/radar/bin/python
A6="--side-csv $F/rib_straight/in/rib_side.csv --seeds 5 --side 0.5 --lev 0.3 --ctx --blocks mean,max:0,1,2:base,bone --nbmode tile"
bash scripts/lane.sh a6k-F2 $B $PY $B/scripts/rs_probe3.py "f_w2|prod/f_w2|$F/rib_straight/in/labels.csv|$F/push90/A/feats_v2|$A6 --nbw 2"
[ -f STOP ] && exit 0
bash scripts/lane.sh a6k-F1b $B $PY $B/scripts/rs_probe2_f.py "f_a6_s5|prod/f_a6_s5|$F/rib_straight/in/labels.csv|$F/push90/A/feats_v2|$A6 --seed0 5"
