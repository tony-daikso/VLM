#!/bin/bash
cd $(dirname $0)
PY=~/miniconda3/envs/radar/bin/python
$PY src/patches.py --config config_interim.yaml --stats > outputs_interim/mask_stats.txt 2>&1
$PY src/probe.py --config config_interim.yaml
$PY src/evaluate.py --config config_interim.yaml --split val
echo INTERIM_DONE
