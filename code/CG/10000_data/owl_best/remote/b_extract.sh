#!/bin/bash
# Tier B stage B on A6000 GPU (shared with bd15: VRAM cap 6 GB, nice 10; does not take A6000_GPU_LOCK, does not preempt):
# per unit/set: wait for CT-rate crops DONE -> rsync crops -> extract with the original script/windows -> keep crops.
# usage: b_extract.sh <set: main|delay>
S=$1; cd ~/Desktop/Tony/VLM/OWL/mw_exp
P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; B=~/Desktop/Tony/VLM/OWL/cg10k_best; CT=daikso@192.168.0.189
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
export OMP_NUM_THREADS=2 MW_ROOT=.
for u in hn fa asc ath rlq bd5; do
  [ -f $B/bemb/$u/$S.DONE ] && continue
  until ssh $CT "test -f OWL/cg10k_best/crops/$u/$S.DONE"; do sleep 120; done
  mkdir -p $B/bcrops/$u/$S $B/bemb/$u/$S
  rsync -a --exclude '*.tmp*' $CT:OWL/cg10k_best/crops/$u/$S/ $B/bcrops/$u/$S/
  ls $B/bcrops/$u/$S | grep '\.npz$' | sed 's/.npz//' > $B/bcrops/$u/$S.ids
  echo "$(date +%T) $u $S crops $(wc -l < $B/bcrops/$u/$S.ids)"
  case $u in
    hn|asc) nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $B/bcrops/$u/$S --out $B/bemb/$u/$S --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 ;;
    fa)     nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $B/bcrops/$u/$S --out $B/bemb/$u/$S --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 --extra air:-700:1000 ;;
    ath)    $P $OF/region_units/scripts/nu_hufeat.py $B/bcrops/$u/$S $B/bemb/ath_hufeat_$S.csv
            nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $B/bcrops/$u/$S --out $B/bemb/$u/$S --ts-model-dir $TS --windows windows.yaml --vram-gb 6 --threads 2 --extra calc:300:800 ;;
    rlq)    nice -n 10 $P $OF/appendicitis_rlq_unit/scripts/rlq_extract.py --crops $B/bcrops/$u/$S --out $B/bemb/$u/$S --ids $B/bcrops/$u/$S.ids --device cuda --config config_final.yaml ;;
    bd5)    nice -n 10 $P $OF/biliary_improve/scripts/bd5_extract.py --crops $B/bcrops/$u/$S --out $B/bemb/$u/$S --ids $B/bcrops/$u/$S.ids --device cuda --config config_final.yaml ;;
  esac > $B/bemb/$u/$S.log 2>&1 && date > $B/bemb/$u/$S.DONE
  echo "$(date +%T) $u $S emb $(ls $B/bemb/$u/$S | wc -l)"
done
echo "B_EXTRACT_DONE $S"
