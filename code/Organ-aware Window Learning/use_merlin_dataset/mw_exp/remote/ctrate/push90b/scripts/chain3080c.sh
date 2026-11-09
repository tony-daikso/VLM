#!/bin/bash
B=/media/daikso/Tony/VLM/data/CT/merlin/push90b; cd $B; PY=/home/daikso/miniconda3/envs/radar/bin/python
while pgrep -f "[c]hain3080.sh" > /dev/null; do [ -f STOP ] && exit 0; sleep 60; done
L=$B/pe/pe_probe_labels_v.csv; FE=/media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/feats
A="--side-csv /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/pe_side.csv --ctx --side 0.5 --lev 0.3 --seeds 5 --nsec 4 --nbmode tile"
bash scripts/lane.sh ct3080c $B $PY $B/scripts/rs_probe2_f.py "pe_tv_s10|pe/prod/pe_tv_s10|$L|$FE|$A --target t_vert --seed0 10"
