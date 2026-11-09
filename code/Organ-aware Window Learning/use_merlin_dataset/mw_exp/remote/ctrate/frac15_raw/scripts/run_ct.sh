#!/bin/bash
# spine15_ft tune config tf2ctx on the CT-rate 3080 (shared with rib_ft2): bf16 autocast, accum 4 (micro-batch 2), VRAM <= 5.5 GB.
# Waits until every tierA id has a crop status. Output runs/s0_tf2ctx (A6000 driver pulls it). stop: touch $R/runs/STOP
R=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw; cd $R; PY=~/miniconda3/envs/radar/bin/python; mkdir -p runs
awk -F, 'NR>1 && $11==0 {print $1}' labels.csv | sort > .tierA_ids.txt
until [ "$(cut -f1 status.tsv | sort -u | comm -12 - .tierA_ids.txt | wc -l)" -ge "$(wc -l < .tierA_ids.txt)" ]; do [ -f runs/STOP ] && exit 0; sleep 120; done
echo "$(date '+%m-%d %H:%M') s0_tf2ctx start" >> logs/run_ct.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels labels.csv --crops crops --out runs/s0_tf2ctx \
  --src ~/OWL/mw_exp/src --epochs 8 --seeds 1 --seed0 0 --train-from 2 --pool lse --ctx --no-val --amp bf16 --accum 4 \
  --vram-gb 5.5 --threads 3 --workers 3 > logs/s0_tf2ctx.log 2>&1
echo "$(date '+%m-%d %H:%M') s0_tf2ctx rc=$? $(grep 'FT train OOF' logs/s0_tf2ctx.log)" >> logs/run_ct.log
