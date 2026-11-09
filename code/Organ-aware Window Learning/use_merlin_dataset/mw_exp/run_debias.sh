#!/bin/bash
cd $(dirname $0)
PY=~/miniconda3/envs/radar/bin/python
$PY src/probe.py --config config_interim.yaml --results-subdir trap --standardize --only E0_trap2 E3_late_trap2
$PY src/probe.py --config config_interim.yaml --results-subdir inlp --standardize --inlp 20 --only E0 E3_late E0_trap2 E3_late_trap2
$PY src/age_shortcut.py --config config_interim.yaml --runs std2/E0 trap/E0_trap trap/E0_trap2 inlp/E0 inlp/E0_trap2 std2/E3_late trap/E3_late_trap trap/E3_late_trap2 inlp/E3_late inlp/E3_late_trap2 --out outputs_interim/results/age_shortcut_debias_val.csv
echo DEBIAS_DONE
