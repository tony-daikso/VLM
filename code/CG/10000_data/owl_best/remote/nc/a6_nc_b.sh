#!/bin/bash
# nc set, tier-B / pelvic / gallstone crop units on the A6000 with the ORIGINAL crop+extract scripts and arguments of
# a6_bpipe.sh / a6_frp.sh / a6_gs.sh (gs5).  Watch mode, batches of BATCH ids (3 mm prep+mask present on CT-rate):
# per unit: CPU crop (nice 19) -> GPU extract in the background while the next unit crops -> crops deleted after extract
# (ath HU scalars are computed from the crops first).  Ends when nifti_3mm_nc/.SYNC_DONE exists and all ids are done.
cd ~/Desktop/Tony/VLM/OWL/mw_exp
P=~/miniconda3/envs/radar/bin/python; OF=outputs_final; B=~/Desktop/Tony/VLM/OWL/cg10k_nc; CT=daikso@192.168.0.189; CG=/media/daikso/CTDATA/cg10k
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
export MW_ROOT=.; D=$B/data/nc; BATCH=${BATCH:-1000}; UNITS="hn fa asc ath rlq bd5 frp gs5"
mkdir -p $D/prep $D/masks $B/bcrops $B/bemb
crop(){ u=$1; I=$2; C=$3; mkdir -p $C
  case $u in
    hn|fa|asc) OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/ru_crop.py $u $I $C 6 $D/prep:$D/masks ;;
    ath)       OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/nu_crop.py ath $I $C 6 $D/prep:$D/masks ;;
    rlq)       OMP_NUM_THREADS=1 nice -n 19 $P $OF/appendicitis_rlq_unit/scripts/rlq_crop.py $I $C 6 $D/prep:$D/masks ;;
    bd5)       OMP_NUM_THREADS=1 nice -n 19 $P $OF/biliary_improve/scripts/bd5_crop.py $I $D/prep $D/masks $C ;;
    frp)       OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/no_crop.py frp $I $C 6 $D/prep:$D/masks ;;
    gs5)       OMP_NUM_THREADS=1 nice -n 19 $P $OF/gallstones_improve/scripts/gs5_crop.py $I $C $D/prep $D/masks 6 ;;
  esac > $C.croplog 2>&1; ls $C | grep '\.npz$' | sed 's/.npz//' > $C.ids; }
ext(){ u=$1; C=$2; k=$3; O=$B/bemb/$u/nc; mkdir -p $O; export OMP_NUM_THREADS=4
  case $u in
    hn|asc) nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 ;;
    fa)     nice -n 10 $P $OF/region_units/scripts/ru_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 --extra air:-700:1000 ;;
    ath)    $P $OF/region_units/scripts/nu_hufeat.py $C $B/bemb/ath_hufeat_nc_b$k.csv
            nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 --extra calc:300:800 ;;
    rlq)    nice -n 10 $P $OF/appendicitis_rlq_unit/scripts/rlq_extract.py --crops $C --out $O --ids $C.ids --device cuda --config config_final.yaml ;;
    bd5)    nice -n 10 $P $OF/biliary_improve/scripts/bd5_extract.py --crops $C --out $O --ids $C.ids --device cuda --config config_final.yaml ;;
    frp)    nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $C --out $O --ts-model-dir $TS --windows windows.yaml --vram-gb 10 --threads 4 ;;
    gs5)    nice -n 10 $P $OF/gallstones_improve/scripts/gs5_extract.py --crops $C --out $O --ids $C.ids --device cuda ;;
  esac > $B/bemb/$u/nc_b$k.log 2>&1 && { cat $C/../nc_b$k.batch >> $B/bemb/$u/nc.processed; rm -rf $C; }
  echo "$(date +%T) extract done $u b$k emb $(ls $O | wc -l)"; }
k=$(ls $B/bcrops/batches 2>/dev/null | wc -l); mkdir -p $B/bcrops/batches
while [ ! -f $B/STOPB ]; do
  ssh $CT "cd $CG; test -f nifti_3mm_nc/.SYNC_DONE && echo SYNC > /dev/stderr; comm -12 <(ls ~/OWL/cg10k_best/data/nc/prep | grep 'npz$' | sed 's/.npz//' | sort) <(ls $CG/owl/nc/masks | grep 'nii.gz$' | sed 's/.nii.gz//' | sort)" > $D/availB.txt 2> $D/syncB.txt
  cat $B/bcrops/batches/* 2>/dev/null | sort -u > $D/doneB.txt
  comm -23 $D/availB.txt $D/doneB.txt | head -n $BATCH > $D/roundB.txt
  if [ ! -s $D/roundB.txt ]; then grep -q SYNC $D/syncB.txt && break; sleep 300; continue; fi
  free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ $free -lt 15 ] && { echo "$(date +%T) disk ${free}G < 15G, wait"; sleep 300; continue; }
  # from the CT-rate SSD copy (ct_feat_nc.sh) to spare the busy CT-rate HDD; only the round's ids
  sed 's/$/.npz/' $D/roundB.txt | rsync -a --files-from=- $CT:OWL/cg10k_best/data/nc/prep/ $D/prep/
  comm -23 $D/roundB.txt <(ls $D/masks | sed 's/.nii.gz//' | sort) | sed 's/$/.nii.gz/' | rsync -a --files-from=- $CT:$CG/owl/nc/masks/ $D/masks/
  echo "$(date +%T) batch $k: $(wc -l < $D/roundB.txt) ids"
  prev=""
  for u in $UNITS; do
    mkdir -p $B/bcrops/$u; C=$B/bcrops/$u/nc_b$k; cp $D/roundB.txt $B/bcrops/$u/nc_b$k.batch
    echo "$(date +%T) crop $u b$k"; crop $u $D/roundB.txt $C; echo "$(date +%T) crop done $u b$k $(wc -l < $C.ids)"
    [ -n "$prev" ] && wait $prev
    ext $u $C $k & prev=$!
  done
  wait $prev; cp $D/roundB.txt $B/bcrops/batches/b$k.txt; k=$((k+1))
done
echo "NC_B_DONE $(date)"
