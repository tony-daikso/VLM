#!/bin/bash
# gallstones GPU-side reproduction on 6 Merlin cases (GB >= 100 vox, i.e. GS6 not involved): gs5 crop+extract and Exp N gs unit
cd ~/Desktop/Tony/VLM/OWL/mw_exp; P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; W=~/Desktop/Tony/VLM/OWL/cg10k_best/gsrepro; mkdir -p $W
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
$P -c "
import pandas as pd, os
ag=pd.concat([pd.read_csv('$OF/analysis_absent_gb/feat_a6.csv'),pd.read_csv('$OF/analysis_absent_gb/feat_ct.csv')])
ids=[s for s in ag[ag.gb_vox>=300].study_id if os.path.exists('outputs/prep/'+s+'.npz') and os.path.exists('$OF/gallstones_improve/gs5/emb/'+s+'.npz') and os.path.exists('$OF/region_units/gallstones/emb/'+s+'.npz')][:6]
open('$W/ids.txt','w').write('\n'.join(ids)+'\n')"
export MW_ROOT=. OMP_NUM_THREADS=2
$P $OF/gallstones_improve/scripts/gs5_crop.py $W/ids.txt $W/crop_gs5 outputs/prep outputs/masks 4 > $W/gs5_crop.log 2>&1
$P $OF/gallstones_improve/scripts/gs5_extract.py --crops $W/crop_gs5 --out $W/emb_gs5 --ids $W/ids.txt --device cuda > $W/gs5_ext.log 2>&1
$P $OF/region_units/scripts/nu_crop.py gs $W/ids.txt $W/crop_gs 4 outputs/prep:outputs/masks > $W/gs_crop.log 2>&1
$P $OF/region_units/scripts/nu_extract.py --crops $W/crop_gs --out $W/emb_gs --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 --extra stone:60:250 calc:300:800 > $W/gs_ext.log 2>&1
$P - <<PY
import numpy as np
for a, b in (("emb_gs5", "$OF/gallstones_improve/gs5/emb"), ("emb_gs", "$OF/region_units/gallstones/emb")):
    d = []
    for i in open("$W/ids.txt").read().split():
        x = np.load(f"$W/{a}/{i}.npz")["emb"].astype(np.float32); y = np.load(f"{b}/{i}.npz")["emb"].astype(np.float32)
        d.append(f"{np.abs(x-y).max():.4f}" if x.shape == y.shape else f"shape {x.shape} {y.shape}")
    print(a, d)
PY
echo GS_REPRO_DONE
