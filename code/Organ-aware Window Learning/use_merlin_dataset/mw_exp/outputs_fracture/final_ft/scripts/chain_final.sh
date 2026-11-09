#!/bin/bash
# wait for spine_ft + pelvis_ft FT_DONE, then final_ft main (rib_ft auto-included if its final files exist)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture
until [ -f spine_ft/runs/FT_DONE ] && [ -f pelvis_ft/runs/FT_DONE ]; do sleep 60; done
cd final_ft
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 nice -n 10 ~/miniconda3/envs/radar/bin/python scripts/final_ft.py --tag main > main.log 2>&1
echo "$(date '+%m-%d %H:%M') final_ft main rc=$?" >> chain.log
