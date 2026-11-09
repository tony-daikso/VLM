#!/bin/bash
# rib_ft2 driver (Spark): 5 folds (train OOF) then full-train model -> val. stop: touch ~/OWL/rib_ft2/runs/main/STOP ; resume: rerun (done folds skipped)
cd ~/OWL/rib_ft2; PY=~/radar_env/bin/python; R=~/OWL/rib_ft
$PY scripts/rib_ft2.py $R/labels.csv $R/roi $R/feats3 runs/main --folds 0,1,2,3,4 --epochs 3 > logs/main_folds.log 2>&1 || exit 1
[ -f runs/main/val_scores_seed0.csv ] || $PY scripts/rib_ft2.py $R/labels.csv $R/roi $R/feats3 runs/main --full --epochs 3 > logs/main_full.log 2>&1
touch runs/main/FT2_DONE
