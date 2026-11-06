#!/bin/bash
# experiment B: full probe with 17 bone units (fracture -> site-wise MIL max), val only, test ids dropped before loading
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
$PY probe.py --config ../config_final.yaml --results-subdir expB --standardize --drop-test --only E0 E3_late
$PY evaluate.py --config ../config_final.yaml --split val --results-subdir expB
$PY age_shortcut.py --config ../config_final.yaml --runs std/E0 std/E3_late expB/E0 expB/E3_late --out ../outputs_final/results/age_shortcut_val_expB.csv
echo EXPB_DONE
