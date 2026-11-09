#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; T=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset295*/nnUNet*)
for i in 1 2 3; do OMP_NUM_THREADS=3 ~/miniconda3/envs/radar/bin/python scripts/pv15_feat.py roi_tier f_295s3 --ts $T --bone --once --maxst 3 --vram-gb 16 --src ~/Desktop/Tony/VLM/OWL/mw_exp/src; done
touch f_295s3/DONE3
OMP_NUM_THREADS=3 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/pv15_probe.py labels.csv f_295s3 runs/t295s3_m400 --ids ids_tier.txt --seeds 3 --side 0.5 --side-csv pel_side.csv --maxinst 400 > logs/t295s3_m400.log 2>&1
