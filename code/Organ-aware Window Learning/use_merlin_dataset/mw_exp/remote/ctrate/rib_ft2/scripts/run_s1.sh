#!/bin/bash
# rib_ft2 seed 1 on the CT-rate 3080 (agent fracture_round2): same recipe as Spark ~/OWL/rib_ft2 (seed 0), 5 folds OOF -> full-train -> val.
# stop: touch /media/daikso/Tony/VLM/data/CT/merlin/rib_ft2/runs/s1/STOP ; resume: rerun this script (done folds skipped)
cd /media/daikso/Tony/VLM/data/CT/merlin/rib_ft2; PY=~/miniconda3/envs/radar/bin/python; R=../rib_ft
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nice -n 5 $PY scripts/rib_ft2.py $R/labels.csv $R/roi $R/feats3 runs/s1 --seed 1 --folds 0,1,2,3,4 --epochs 3 --workers 4 --src ~/OWL/mw_exp/src > logs/s1_folds.log 2>&1 || exit 1
[ -f runs/s1/val_scores_seed1.csv ] || nice -n 5 $PY scripts/rib_ft2.py $R/labels.csv $R/roi $R/feats3 runs/s1 --seed 1 --full --epochs 3 --workers 4 --src ~/OWL/mw_exp/src > logs/s1_full.log 2>&1
touch runs/s1/FT2_DONE
