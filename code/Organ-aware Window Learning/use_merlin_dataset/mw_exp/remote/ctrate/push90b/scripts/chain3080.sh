#!/bin/bash
B=/media/daikso/Tony/VLM/data/CT/merlin/push90b; cd $B; PY=/home/daikso/miniconda3/envs/radar/bin/python
while pgrep -f "[r]s_probe2_f.py .*pe/prod/pe_tv_s0 " > /dev/null; do [ -f STOP ] && exit 0; sleep 60; done
echo "$(date "+%m-%d %H:%M") ct3080 pe_tv_s0 python exited, summary: $(ls pe/prod/pe_tv_s0/summary.json 2>/dev/null) $(grep PROBE logs/pe_tv_s0.log | tail -1 | cut -c1-200)" >> logs/push90b.log
L=$B/pe/pe_probe_labels_v.csv; FE=/media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/feats
A="--side-csv /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/pe_side.csv --ctx --side 0.5 --lev 0.3 --seeds 5 --nsec 4 --nbmode tile"
bash scripts/lane.sh ct3080b $B $PY $B/scripts/rs_probe2_f.py "pe_tv_s0|pe/prod/pe_tv_s0|$L|$FE|$A --target t_vert --seed0 0" "pe_y_s0|pe/prod/pe_y_s0|$L|$FE|$A --target y --seed0 0"
