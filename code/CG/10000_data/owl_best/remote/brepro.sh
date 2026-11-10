#!/bin/bash
# Tier B reproduction check on A6000: crop + extract 8 Merlin cases per unit with the original scripts and compare
# with the stored Merlin embeddings (GPU shared with bd15, VRAM cap 6 GB, nice 10).
cd ~/Desktop/Tony/VLM/OWL/mw_exp
P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; W=~/Desktop/Tony/VLM/OWL/cg10k_best/brepro; mkdir -p $W
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
ls $OF/region_units/hydronephrosis/emb | head -400 | sed 's/.npz//' | while read i; do [ -f outputs/prep/$i.npz ] && [ -f outputs/masks/$i.nii.gz ] && echo $i; done | head -8 > $W/ids.txt
export OMP_NUM_THREADS=2 MW_ROOT=.
for u in hn fa asc; do $P $OF/region_units/scripts/ru_crop.py $u $W/ids.txt $W/crop_$u 4 outputs/prep:outputs/masks > $W/crop_$u.log 2>&1; done
$P $OF/region_units/scripts/nu_crop.py ath $W/ids.txt $W/crop_ath 4 outputs/prep:outputs/masks > $W/crop_ath.log 2>&1
$P $OF/appendicitis_rlq_unit/scripts/rlq_crop.py $W/ids.txt $W/crop_rlq 4 outputs/prep:outputs/masks > $W/crop_rlq.log 2>&1
$P $OF/biliary_improve/scripts/bd5_crop.py $W/ids.txt outputs/prep outputs/masks $W/crop_bd5 > $W/crop_bd5.log 2>&1
X(){ u=$1; shift; nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $W/crop_$u --out $W/emb_$u --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 "$@" > $W/ext_$u.log 2>&1; }
X hn; X fa --extra air:-700:1000; X asc
nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $W/crop_ath --out $W/emb_ath --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 --extra calc:300:800 > $W/ext_ath.log 2>&1
nice -n 10 $P $OF/appendicitis_rlq_unit/scripts/rlq_extract.py --crops $W/crop_rlq --out $W/emb_rlq --ids $W/ids.txt --device cuda --config config_final.yaml > $W/ext_rlq.log 2>&1
nice -n 10 $P $OF/biliary_improve/scripts/bd5_extract.py --crops $W/crop_bd5 --out $W/emb_bd5 --ids $W/ids.txt --device cuda --config config_final.yaml > $W/ext_bd5.log 2>&1
$P - <<'PY'
import numpy as np, os
W = os.path.expanduser("~/Desktop/Tony/VLM/OWL/cg10k_best/brepro/"); OF = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/")
ref = {"hn": "region_units/hydronephrosis/emb", "fa": "region_units/free_air/emb", "asc": "region_units/ascites/emb",
       "ath": "region_units/atherosclerosis/emb", "rlq": "appendicitis_rlq_unit/emb_rlq", "bd5": "biliary_improve/emb_porta"}
ids = open(W + "ids.txt").read().split()
for u, r in ref.items():
    d = []
    for i in ids:
        a, b = W + f"emb_{u}/{i}.npz", OF + f"{r}/{i}.npz"
        if not (os.path.exists(a) and os.path.exists(b)): d.append("missing"); continue
        za, zb = np.load(a), np.load(b); k = "emb" if "emb" in zb.files else zb.files[0]
        x, y = za[k].astype(np.float32), zb[k].astype(np.float32)
        d.append(f"{np.nanmax(np.abs(x - y)):.4f}" if x.shape == y.shape else f"shape {x.shape} vs {y.shape}")
    print(u, d, flush=True)
PY
echo BREPRO_DONE
