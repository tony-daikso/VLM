#!/bin/bash
# GS6 steps 1-2 for CG10k (original gs6_boxes.py / gs6_rawcrop.py): GB-missing/tiny cases (3 mm GB < 100 vox) ->
# world box (liver bbox U expected fossa, +45 mm) -> crop of the CG NIfTI (3 mm source) -> TS-identical 1.5 mm resampling.
P=~/miniconda3/envs/radar/bin/python; B=~/OWL/cg10k_best; CG=/media/daikso/CTDATA/cg10k
for S in main delay; do
  nif=$CG/nifti_3mm; [ $S = delay ] && nif=$CG/nifti_3mm_delay
  mkdir -p $B/gs6/$S/raw15
  [ -f $B/gs6/$S/boxes_all.csv ] || [ -f $B/gs6/$S/boxes.csv ] || nice -n 19 $P $B/gs6_boxes.py $CG/owl_feat/$S/ids_ok.txt $B/data/$S/prep $B/gs6/$S/boxes.csv
  [ -f $B/gs6/$S/boxes_all.csv ] || mv $B/gs6/$S/boxes.csv $B/gs6/$S/boxes_all.csv
  $P -c "import pandas as pd; b=pd.read_csv('$B/gs6/$S/boxes_all.csv'); b[b.gb_vox3 < 100].to_csv('$B/gs6/$S/boxes.csv', index=False)"   # GS6 = 3 mm GB < 100 vox only
  nice -n 19 $P $B/gs6_rawcrop.py $B/gs6/$S/boxes.csv $nif $B/gs6/$S/raw15 6 > $B/gs6/$S/rawcrop.log 2>&1
  echo "$S boxes $(grep -c ",box,\|,full," $B/gs6/$S/boxes.csv) crops $(ls $B/gs6/$S/raw15 | wc -l)"
done
date > $B/gs6/PREP_DONE
