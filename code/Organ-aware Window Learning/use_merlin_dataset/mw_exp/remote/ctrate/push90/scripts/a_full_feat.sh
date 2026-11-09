#!/bin/bash
# push90 A: full v2 features (base+bone, stages 0-3) on the CT-rate 3080 (bf16 encoder), part 4/5. stop: touch A/feats_v2/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/push90
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/rs_feat2.py /media/daikso/Tony/VLM/data/CT/merlin/rib_straight/tiles A/feats_v2 --ids A/ids_full.txt --part 4/5 --stages 0,1,2,3 --wins base,bone --vram-gb 5 --bs 8 --amp bf16 --src ~/OWL/mw_exp/src --ts ~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres > logs/a_feat_v2.log 2>&1
echo "$(date '+%m-%d %H:%M') A v2 feats part 4 done rc=$?" >> logs/push90.log
