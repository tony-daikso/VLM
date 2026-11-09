#!/bin/bash
cd $(dirname $0)
PY=~/miniconda3/envs/radar/bin/python
$PY src/probe.py --config config_interim.yaml --results-subdir trap --standardize --only E0_trap E3_late_trap
$PY src/age_shortcut.py --config config_interim.yaml --runs std2/E0 trap/E0_trap std2/E3_late trap/E3_late_trap
echo TRAP_DONE
