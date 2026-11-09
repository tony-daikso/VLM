#!/bin/bash
cd $(dirname $0)
PY=~/miniconda3/envs/radar/bin/python
$PY src/probe.py --config config_interim.yaml --results-subdir gate --only E3_slow E3_2stage
$PY src/evaluate.py --config config_interim.yaml --split val --results-subdir gate
echo GATE_DONE
$PY src/probe.py --config config_interim.yaml --results-subdir std --standardize
$PY src/evaluate.py --config config_interim.yaml --split val --results-subdir std
echo STD_DONE
