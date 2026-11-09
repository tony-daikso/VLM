#!/bin/bash
# push90 B: PE frozen features on the CT-rate 3080, following the crop run (watch mode). stop: touch /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/feats/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15; mkdir -p feats
( until grep -qE 'CROP_DONE|abort' logs/crop.log; do sleep 120; done; touch CROPS_DONE ) &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/pe_feat.py crops feats --ids ids_trval.txt --src ~/OWL/mw_exp/src --amp bf16 --vram-gb 6 --watch --done-flag /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/CROPS_DONE >> logs/feat.log 2>&1
echo "$(date '+%m-%d %H:%M') PE feat exit rc=$?" >> logs/feat.log
