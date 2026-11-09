#!/bin/bash
cd $(dirname $0)
PY=~/miniconda3/envs/radar/bin/python
$PY src/probe.py --config config_interim.yaml --results-subdir std2 --standardize --only E3_late E0_rf
$PY src/evaluate.py --config config_interim.yaml --split val --results-subdir std2
echo STD2_DONE
