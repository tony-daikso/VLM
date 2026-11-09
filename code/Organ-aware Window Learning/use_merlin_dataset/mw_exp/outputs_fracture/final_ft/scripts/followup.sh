#!/bin/bash
# follow-up (A6000): (1) spine seed 2 -> spine_ft/runs/final3_tf2lse (mean of s0,s1,s2) -> final_ft --tag main_spine3seed
# (2) whenever rib_ft final files appear (rib_ft/runs/best_cfg.txt + final_*/val_scores.csv, or rib_ft/oof_train.csv + val_scores.csv)
#     -> final_ft --tag with_rib (once). stop: touch outputs_fracture/spine_ft/runs/STOP (training) / kill this script.
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; PY=~/miniconda3/envs/radar/bin/python; S=$F/spine_ft
log(){ echo "$(date '+%m-%d %H:%M') $*" >> $F/final_ft/chain.log; }
cd $S
if [ ! -f runs/s2_tf2lse/summary.json ]; then
  log "spine s2 start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac_ft.py --branch spine --labels in/labels.csv \
    --packs in/pack_a6.npz in/pack_ct.npz --out runs/s2_tf2lse --seeds 1 --seed0 2 --train-from 2 --pool lse --epochs 8 > logs/s2_tf2lse.log 2>&1 &
  TP=$!
fi
ribdone=0
while true; do
  if [ $ribdone = 0 ] && { [ -f $F/rib_ft/runs/best_cfg.txt ] && ls $F/rib_ft/runs/final_*/val_scores.csv >/dev/null 2>&1 || [ -f $F/rib_ft/val_scores.csv ]; }; then
    sleep 120; log "rib_ft found -> final_ft with_rib"
    (cd $F/final_ft && OMP_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag with_rib > with_rib.log 2>&1); log "with_rib rc=$?"; ribdone=1
  fi
  if [ -n "$TP" ] && ! kill -0 $TP 2>/dev/null; then
    TP=""
    if [ -f runs/s2_tf2lse/summary.json ]; then
      $PY -c "
import pandas as pd, os
os.makedirs('runs/final3_tf2lse', exist_ok=True)
for f in ('oof_train.csv','val_scores.csv'):
    ds=[pd.read_csv(f'runs/s{k}_tf2lse/{f}') for k in (0,1,2)]; o=ds[0].copy(); o['ft']=sum(d['ft'] for d in ds)/3; o.to_csv(f'runs/final3_tf2lse/{f}',index=False)"
      log "spine s2 done -> final_ft main_spine3seed"
      (cd $F/final_ft && OMP_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag main_spine3seed --spine $S/runs/final3_tf2lse > main_spine3seed.log 2>&1); log "main_spine3seed rc=$?"
      [ $ribdone = 1 ] && (cd $F/final_ft && OMP_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag with_rib_spine3seed --spine $S/runs/final3_tf2lse > with_rib_spine3seed.log 2>&1)
    else log "spine s2 failed/stopped"; fi
  fi
  [ -z "$TP" ] && [ $ribdone = 1 ] && break
  sleep 60
done
