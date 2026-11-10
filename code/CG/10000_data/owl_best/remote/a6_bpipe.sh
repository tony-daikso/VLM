#!/bin/bash
# Tier B on A6000 (12:04 A6000 GPU idle): CG prep/masks rsync'd from CT-rate SSD (internal host) ->
# per unit: CPU crop (original scripts, 6 procs, nice 19) -> GPU extract in the background (original scripts/windows,
# VRAM cap 10 GB) while the next unit crops.  Crops kept (ath HU scalars need them).  usage: a6_bpipe.sh
cd ~/Desktop/Tony/VLM/OWL/mw_exp
P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; B=~/Desktop/Tony/VLM/OWL/cg10k_best; CT=daikso@192.168.0.189
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
export OMP_NUM_THREADS=1 MW_ROOT=.
for S in main delay; do
  mkdir -p $B/data/$S; rsync -a $CT:OWL/cg10k_best/data/$S/ $B/data/$S/
  ls $B/data/$S/prep | sed 's/.npz//' | sort > $B/data/$S/ids.txt; echo "$(date +%T) data $S $(wc -l < $B/data/$S/ids.txt)"
done
crop(){ u=$1; S=$2; D=$B/data/$S; I=$D/ids.txt; C=$B/bcrops/$u/$S; mkdir -p $C
  case $u in
    hn|fa|asc) nice -n 19 $P $OF/region_units/scripts/ru_crop.py $u $I $C 6 $D/prep:$D/masks ;;
    ath)       nice -n 19 $P $OF/region_units/scripts/nu_crop.py ath $I $C 6 $D/prep:$D/masks ;;
    rlq)       nice -n 19 $P $OF/appendicitis_rlq_unit/scripts/rlq_crop.py $I $C 6 $D/prep:$D/masks ;;
    bd5)       nice -n 19 $P $OF/biliary_improve/scripts/bd5_crop.py $I $D/prep $D/masks $C ;;
  esac > $B/bcrops/$u/$S.croplog 2>&1; ls $C | grep '\.npz$' | sed 's/.npz//' > $B/bcrops/$u/$S.ids; }
ext(){ u=$1; S=$2; C=$B/bcrops/$u/$S; O=$B/bemb/$u/$S; mkdir -p $O; export OMP_NUM_THREADS=4
  case $u in
    hn|asc) nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 ;;
    fa)     nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 --extra air:-700:1000 ;;
    ath)    $P $OF/region_units/scripts/nu_hufeat.py $C $B/bemb/ath_hufeat_$S.csv
            nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 --extra calc:300:800 ;;
    rlq)    nice -n 10 $P $OF/appendicitis_rlq_unit/scripts/rlq_extract.py --crops $C --out $O --ids $B/bcrops/$u/$S.ids --device cuda --config config_final.yaml ;;
    bd5)    nice -n 10 $P $OF/biliary_improve/scripts/bd5_extract.py --crops $C --out $O --ids $B/bcrops/$u/$S.ids --device cuda --config config_final.yaml ;;
  esac > $B/bemb/$u/$S.log 2>&1 && date > $B/bemb/$u/$S.DONE; echo "$(date +%T) extract done $u $S $(ls $O | wc -l)"; }
prev=""
for S in main delay; do for u in hn fa asc ath rlq bd5; do
  [ -f $B/bemb/$u/$S.DONE ] && continue
  echo "$(date +%T) crop $u $S"; crop $u $S; echo "$(date +%T) crop done $u $S $(wc -l < $B/bcrops/$u/$S.ids)"
  [ -n "$prev" ] && wait $prev
  ext $u $S & prev=$!
done; done
wait; echo "A6_BPIPE_DONE $(date)"
