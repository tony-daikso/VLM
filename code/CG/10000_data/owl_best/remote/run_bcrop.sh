#!/bin/bash
# Tier B stage A (CPU crops) for CG10k, after the tier-A features of the same set are done. nice 19, 4 procs.
S=$1; NP=${2:-4}; P=~/miniconda3/envs/radar/bin/python; B=~/OWL/cg10k_best; SRC=$B/bsrc
D=$B/data/$S; I=/media/daikso/CTDATA/cg10k/owl_feat/$S/ids_ok.txt; C=$B/crops
while [ ! -f /media/daikso/CTDATA/cg10k/owl_feat/$S/ALL_DONE ]; do sleep 120; done
export OMP_NUM_THREADS=1
r(){ n=$1; shift; [ -f $C/$n/$S.DONE ] && return; mkdir -p $C/$n/$S; echo "$(date +%T) start $n"; nice -n 19 "$@" > $C/$n/$S.log 2>&1 && date > $C/$n/$S.DONE; echo "$(date +%T) end $n"; }
r hn   $P $SRC/region_units/scripts/ru_crop.py hn  $I $C/hn/$S  $NP $D/prep:$D/masks
r fa   $P $SRC/region_units/scripts/ru_crop.py fa  $I $C/fa/$S  $NP $D/prep:$D/masks
r asc  $P $SRC/region_units/scripts/ru_crop.py asc $I $C/asc/$S $NP $D/prep:$D/masks
r ath  $P $SRC/region_units/scripts/nu_crop.py ath $I $C/ath/$S $NP $D/prep:$D/masks
r rlq  $P $SRC/appendicitis_rlq_unit/scripts/rlq_crop.py $I $C/rlq/$S $NP $D/prep:$D/masks
r bd5  $P $SRC/biliary_improve/scripts/bd5_crop.py $I $D/prep $D/masks $C/bd5/$S
echo "$(date) BCROP DONE $S"
