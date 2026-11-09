#!/bin/bash
# push90 PE crops (CT-rate CPU, 12 procs). stop: touch /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/STOP ; resume: rerun (done ids skipped)
cd /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15
nice -n 10 ~/miniconda3/envs/radar/bin/python scripts/pe15_crop.py ids_trval.txt /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15 12 /media/daikso/Tony/VLM/data/CT/merlin/vert15_seg15 ~/OWL/mw_exp/outputs/masks /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/masks3 >> logs/crop.log 2>&1
