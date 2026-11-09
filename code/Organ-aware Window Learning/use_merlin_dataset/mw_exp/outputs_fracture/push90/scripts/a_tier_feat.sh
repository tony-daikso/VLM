#!/bin/bash
# push90 A: tier feature extraction (A6000 GPU, 2 parts). stop: touch A/feats_tier/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90
for p in 0 1; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/rs_feat2.py ../rib_straight/tiles A/feats_tier --ids ../rib_straight/in/ids_tier.txt --part $p/2 --vram-gb 14 > logs/a_feat_tier_p$p.log 2>&1 &
done
wait; echo "$(date '+%m-%d %H:%M') A tier feats done" >> logs/push90.log
