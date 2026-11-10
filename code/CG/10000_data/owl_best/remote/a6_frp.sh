#!/bin/bash
# Fracture Exp O pelvic-ring unit (frp, K=4 crops 48^3, 5 windows) for CG10k on A6000, after a6_bpipe.sh.
cd ~/Desktop/Tony/VLM/OWL/mw_exp; P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; B=~/Desktop/Tony/VLM/OWL/cg10k_best
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
# (13:36) start immediately: GPU has spare capacity while bpipe crops are CPU-bound
export MW_ROOT=.
for S in main delay; do
  [ -f $B/bemb/frp/$S.DONE ] && continue
  D=$B/data/$S; C=$B/bcrops/frp/$S; O=$B/bemb/frp/$S; mkdir -p $C $O
  OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/no_crop.py frp $D/ids.txt $C 4 $D/prep:$D/masks > $B/bcrops/frp/$S.croplog 2>&1
  OMP_NUM_THREADS=4 nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 > $B/bemb/frp/$S.log 2>&1 && date > $B/bemb/frp/$S.DONE
  echo "$(date +%T) frp $S emb $(ls $O | wc -l)"
done
echo FRP_DONE
