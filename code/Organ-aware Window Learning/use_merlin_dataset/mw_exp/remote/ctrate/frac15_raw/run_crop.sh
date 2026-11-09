#!/bin/bash
# spine15_ft crops (CT-rate CPU). stop: touch /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/STOP ; resume: rerun this script (done ids skipped)
cd /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw
nice -n 10 ~/miniconda3/envs/radar/bin/python scripts/spine15_crop.py ids_prio.txt /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw 12 /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/../vert15_seg15 ~/OWL/mw_exp/outputs/masks /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/masks3 >> logs/crop.log 2>&1
