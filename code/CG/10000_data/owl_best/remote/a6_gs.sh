#!/bin/bash
# Gallstones Exp N (N_U = G5c + GB/GB-fossa unit U) inputs for CG10k on A6000, original scripts only
# (gs5 crop+extract reproduced 0.0 diff, Exp N gs unit 5e-4 diff on 6 Merlin cases):
#  1 gs5_crop (CPU) -> gs5_extract (GPU)                         -> gbx head inputs
#  2 GS6: CT-rate boxes+raw 1.5 mm crops -> gs6_seg (Dataset291, GPU) for 3 mm GB < 100 vox cases
#  3 nu_crop gs with GS6_SEG (single pass: cases with a GS6 seg get the GS6 GB, others unchanged) -> nu_extract (+stone, calc)
cd ~/Desktop/Tony/VLM/OWL/mw_exp; P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; B=~/Desktop/Tony/VLM/OWL/cg10k_best; CT=daikso@192.168.0.189
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
export MW_ROOT=.
for S in main delay; do D=$B/data/$S
  mkdir -p $B/bcrops/gs5/$S $B/bemb/gs5/$S
  [ -f $B/bemb/gs5/$S.DONE ] || { OMP_NUM_THREADS=1 nice -n 19 $P $OF/gallstones_improve/scripts/gs5_crop.py $D/ids.txt $B/bcrops/gs5/$S $D/prep $D/masks 4 > $B/bcrops/gs5/$S.log 2>&1
    ls $B/bcrops/gs5/$S | grep npz$ | sed 's/.npz//' > $B/bcrops/gs5/$S.ids
    OMP_NUM_THREADS=4 nice -n 10 $P $OF/gallstones_improve/scripts/gs5_extract.py --crops $B/bcrops/gs5/$S --out $B/bemb/gs5/$S --ids $B/bcrops/gs5/$S.ids --device cuda > $B/bemb/gs5/$S.log 2>&1 && date > $B/bemb/gs5/$S.DONE; }
  echo "$(date +%T) gs5 $S $(ls $B/bemb/gs5/$S | wc -l)"
done
until ssh $CT "test -f OWL/cg10k_best/gs6/PREP_DONE"; do sleep 120; done
for S in main delay; do
  mkdir -p $B/gs6/$S/raw15 $B/gs6/$S/seg; rsync -a $CT:OWL/cg10k_best/gs6/$S/ $B/gs6/$S/ --exclude seg
  touch $B/gs6/$S/raw15/DONE_SENDING
  [ -f $B/gs6/$S/SEG_DONE ] || { OMP_NUM_THREADS=2 nice -n 10 $P /home/advantech/Desktop/Tony/VLM/OWL/cg10k_best/gs6_seg_cg.py --inbox $B/gs6/$S/raw15 --out $B/gs6/$S/seg --device cuda > $B/gs6/$S/seg.log 2>&1 && date > $B/gs6/$S/SEG_DONE; }
  echo "$(date +%T) gs6 seg $S $(ls $B/gs6/$S/seg | grep -c nii)"
  D=$B/data/$S; mkdir -p $B/bcrops/gs/$S $B/bemb/gs/$S
  [ -f $B/bemb/gs/$S.DONE ] || { GS6_SEG=$B/gs6/$S/seg OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/nu_crop.py gs $D/ids.txt $B/bcrops/gs/$S 4 $D/prep:$D/masks > $B/bcrops/gs/$S.log 2>&1
    OMP_NUM_THREADS=4 nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $B/bcrops/gs/$S --out $B/bemb/gs/$S --ts-model-dir $TS --windows windows.yaml --vram-gb 8 --threads 4 --extra stone:60:250 calc:300:800 > $B/bemb/gs/$S.log 2>&1 && date > $B/bemb/gs/$S.DONE; }
  echo "$(date +%T) gs unit $S $(ls $B/bemb/gs/$S | wc -l)"
done
echo A6_GS_DONE
