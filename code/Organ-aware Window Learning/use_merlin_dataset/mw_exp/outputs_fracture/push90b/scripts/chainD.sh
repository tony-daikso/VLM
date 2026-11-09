#!/bin/bash
# after chainC (F1 resume) ends: f_a6_s10 folds 0,1,2 here (folds 3,4 on Spark, merged later by rsync + resume pass)
F=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; B=$F/push90b; cd $B; PY=/home/advantech/miniconda3/envs/radar/bin/python
while pgrep -f "[c]hainC.sh" > /dev/null; do [ -f STOP ] && exit 0; sleep 60; done; [ -f STOP ] && exit 0
A6="--side-csv $F/rib_straight/in/rib_side.csv --seeds 5 --side 0.5 --lev 0.3 --ctx --blocks mean,max:0,1,2:base,bone --nbmode tile --seed0 10"
echo "$(date "+%m-%d %H:%M") a6k-R f_a6_s10 folds 0,1,2 start" >> logs/push90b.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY $B/scripts/rs_probe2_f.py $F/rib_straight/in/labels.csv $F/push90/A/feats_v2 $B/prod/f_a6_s10 $A6 --folds 0,1,2 >> logs/f_a6_s10.log 2>&1
echo "$(date "+%m-%d %H:%M") a6k-R f_a6_s10 folds 0,1,2 rc=$? $(ls prod/f_a6_s10 | tr '\n' ' ')" >> logs/push90b.log
