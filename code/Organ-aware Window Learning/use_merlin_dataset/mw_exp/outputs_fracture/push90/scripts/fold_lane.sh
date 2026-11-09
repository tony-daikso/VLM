#!/bin/bash
# push90 fold-parallel full OOF lane: rs_probe2_f.py on the given folds of run NAME (2 tries); when all 5 score_fold files
# exist, the summary pass runs (all folds resumed). usage: fold_lane.sh BASE PY LABELS FEATS SIDE NAME FOLDS "args"
# stop: touch BASE/STOP
B=$1; PY=$2; LAB=$3; FE=$4; SIDE=$5; n=$6; FO=$7; AR=$8; cd $B; mkdir -p prod logs
for try in 1 2; do
  miss=0; for k in ${FO//,/ }; do [ -f prod/$n/score_fold$k.csv ] || miss=1; done
  [ $miss = 0 ] && break; [ -f STOP ] && exit 0
  echo "$(date "+%m-%d %H:%M") $(hostname | cut -c1-8) full $n folds $FO start try $try" >> logs/push90.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rs_probe2_f.py $LAB $FE prod/$n --side-csv $SIDE --folds $FO $AR >> logs/full_${n}_f${FO//,/}.log 2>&1
  echo "$(date "+%m-%d %H:%M") $(hostname | cut -c1-8) full $n folds $FO rc=$?" >> logs/push90.log
done
