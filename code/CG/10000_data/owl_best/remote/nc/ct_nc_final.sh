#!/bin/bash
# CT-rate, nc set, after the nc 3 mm OWL run is complete (run_nc.sh "ALL DONE" -> owl/probs_nc.csv):
#  features for the remaining ids (ct_feat_nc.sh part 1) -> merged feature csvs, OWL matrix (build_cg_matrix.py),
#  fracture bone rows (cg_frac_rows.py, prefix D_ = nc slot of the cg10k_nc sandbox), GS6 boxes + 1.5 mm raw crops.
B=~/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python; CG=/media/daikso/CTDATA/cg10k; F=$CG/owl_feat/nc
cd $B
until grep -q "ALL DONE" ~/OWL/cg10k_infer/logs/nc_run.log && [ -f $F/part0/ALL_DONE ]; do sleep 120; done
echo "$(date +%T) nc 3 mm run done"
bash $B/ct_feat_nc.sh 1 8 > logs/feat_nc1.log 2>&1
mkdir -p $F/all
$P - <<PY
import pandas as pd, glob, os
F = "$F"
fs = sorted({os.path.basename(f) for f in glob.glob(F + "/part*/*.csv")})
for f in fs:
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(F + f"/part*/{f}"))], ignore_index=True)
    k = [c for c in d.columns if c.lower() in ("study_id", "id", "examid")][0]
    d = d.drop_duplicates(k, keep="last"); d.to_csv(F + "/all/" + f, index=False); print(f, d.shape)
PY
cat $F/part*/ids_ok.txt | sort -u > $F/all/ids_ok.txt
nice -n 19 $P build_cg_matrix.py $CG/owl/nc mat/cg_nc.npz > logs/mat_nc.log 2>&1
nice -n 19 $P cg_frac_rows.py $CG/owl/nc/emb_bones D_ mat/cg_nc_bones.npz > logs/frac_rows_nc.log 2>&1
# GS6 steps 1-2 (as run_gs6_prep.sh): 3 mm GB < 100 vox -> box -> 1.5 mm raw crop of the nc NIfTI
mkdir -p gs6/nc/raw15
nice -n 19 $P gs6_boxes.py $F/all/ids_ok.txt data/nc/prep gs6/nc/boxes_all.csv
$P -c "import pandas as pd; b=pd.read_csv('gs6/nc/boxes_all.csv'); b[b.gb_vox3 < 100].to_csv('gs6/nc/boxes.csv', index=False)"
nice -n 19 $P gs6_rawcrop.py gs6/nc/boxes.csv $CG/nifti_3mm_nc gs6/nc/raw15 6 > gs6/nc/rawcrop.log 2>&1
echo "nc boxes $(grep -c ',box,\|,full,' gs6/nc/boxes.csv) crops $(ls gs6/nc/raw15 | wc -l)"
date > $B/NC_CT_FINAL_DONE; echo NC_CT_FINAL_DONE
