#!/bin/bash
# push90 A: full v2 features (base+bone, stages 0-3) on Spark, parts of 5 as args. stop: touch A/feats_v2/STOP
cd ~/Desktop/Tony/OWL/push90
for p in "$@"; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 ~/radar_env/bin/python scripts/rs_feat2.py ~/Desktop/Tony/OWL/rib_straight/tiles A/feats_v2 --ids A/ids_full.txt --part $p/5 --stages 0,1,2,3 --wins base,bone --vram-gb 16 --src ~/Desktop/Tony/OWL/mw_exp/src > logs/a_feat_v2_p$p.log 2>&1 &
done
wait; echo "$(date "+%m-%d %H:%M") A v2 feats parts $* done" >> logs/push90.log
