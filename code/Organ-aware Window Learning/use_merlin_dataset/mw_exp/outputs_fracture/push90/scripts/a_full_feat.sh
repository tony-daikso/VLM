#!/bin/bash
# push90 A: full v2 feature extraction (base+bone, stages 0-3) on A6000, parts given as args (of 5). stop: touch A/feats_v2/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90
for p in "$@"; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/rs_feat2.py ../rib_straight/tiles A/feats_v2 --ids A/ids_full.txt --part $p/5 --stages 0,1,2,3 --wins base,bone --vram-gb 10 > logs/a_feat_v2_p$p.log 2>&1 &
done
wait; echo "$(date "+%m-%d %H:%M") A v2 feats parts $* done" >> logs/push90.log
