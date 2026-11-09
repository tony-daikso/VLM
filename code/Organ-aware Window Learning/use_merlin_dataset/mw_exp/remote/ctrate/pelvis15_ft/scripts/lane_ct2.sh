#!/bin/bash
# t297sm folds 3,4 on the 3080 (bf16) to speed up the A6000 t297sm tier run; fold npz files are pushed to the A6000 run dir
# (A6000 skips folds whose fold file exists). agent pelvis15_ft 11-08. stop: touch runs/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft; P=../pelvis15; TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset297_*/nnUNet*)
echo "$(date "+%m-%d %H:%M") t297sm_f34 start" >> logs/tier.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/pv15_ft.py --labels $P/labels.csv --roi $P/roi \
  --out runs/t297sm_f34 --ts $TS --src ~/OWL/mw_exp/src --tune --pool-ids $P/ids_tier.txt --side-csv $P/pel_side.csv --fixed --maxshape 176,144,240 \
  --threads 3 --side 0.5 --mir --amp bf16 --vram-gb 9 --workers 4 --folds 3,4 >> logs/t297sm_f34.log 2>&1
echo "$(date "+%m-%d %H:%M") t297sm_f34 rc=$?" >> logs/tier.log
