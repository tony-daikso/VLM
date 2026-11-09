#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp
PY=~/miniconda3/envs/radar/bin/python
$PY src/probe.py --config config_final.yaml --results-subdir std --standardize --only E0 E1_soft_tissue E1_liver E1_narrow_abd E1_bone E2 E3 E3_late E4 E0_rf E0_trap E3_late_trap E0_trap2 E3_late_trap2
$PY src/evaluate.py --config config_final.yaml --split val --results-subdir std
$PY src/age_shortcut.py --config config_final.yaml --runs std/E0 std/E3_late std/E0_trap std/E3_late_trap std/E0_trap2 std/E3_late_trap2 --out outputs_final/results/age_shortcut_val.csv
echo FINAL_DONE
