#!/bin/bash
# CT-rate 3080 queue after s0_tf2ctx: run seed $2 of config $1 (bf16, accum 4, <=5.5 GB, per-epoch ckpt) unless the A6000
# already produced it. usage: run_ct2.sh tf2ctx|tf2lse SEED. stop: touch $R/runs/STOP
R=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw; cd $R; PY=~/miniconda3/envs/radar/bin/python
C=$1; K=$2; X=""; [ "$C" = tf2ctx ] && X="--ctx"
until [ -f runs/s0_tf2ctx/summary.json ]; do [ -f runs/STOP ] && exit 0; sleep 60; done
echo "$(date '+%m-%d %H:%M') s${K}_$C start" >> logs/run_ct.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels labels.csv --crops crops --out runs/s${K}_$C \
  --src ~/OWL/mw_exp/src --epochs 8 --seeds 1 --seed0 $K --train-from 2 --pool lse $X --no-val --amp bf16 --accum 4 \
  --vram-gb 5.5 --threads 3 --workers 3 > logs/s${K}_$C.log 2>&1
echo "$(date '+%m-%d %H:%M') s${K}_$C rc=$? $(grep 'FT train OOF' logs/s${K}_$C.log)" >> logs/run_ct.log
