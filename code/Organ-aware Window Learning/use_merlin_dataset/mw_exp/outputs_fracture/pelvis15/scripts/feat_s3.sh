#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; T=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset292*/nnUNet*)
for i in 1 2 3; do OMP_NUM_THREADS=3 ~/miniconda3/envs/radar/bin/python scripts/pv15_feat.py roi_tier f_292s3 --ts $T --bone --once --maxst 3 --vram-gb 14 --src ~/Desktop/Tony/VLM/OWL/mw_exp/src; done
touch f_292s3/DONE3
