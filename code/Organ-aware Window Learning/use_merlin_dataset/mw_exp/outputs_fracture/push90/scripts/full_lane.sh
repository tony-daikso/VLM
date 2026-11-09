#!/bin/bash
# push90 full-OOF lane (rs_probe2; all train 5 sitewise folds OOF + val fold-mean; per-fold resume; 2 tries per run, a run
# failing twice stops the lane). usage: full_lane.sh BASE PY LABELS FEATS SIDE "name|args" ...   stop: touch BASE/STOP
B=$1; PY=$2; LAB=$3; FE=$4; SIDE=$5; shift 5; cd $B; mkdir -p prod logs
for spec in "$@"; do
  n=${spec%%|*}; ar=${spec#*|}; ok=0
  for try in 1 2; do
    [ -f prod/$n/summary.json ] && { ok=1; break; }; [ -f STOP ] && exit 0
    echo "$(date "+%m-%d %H:%M") $(hostname | cut -c1-8) full $n start try $try" >> logs/push90.log
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rs_probe2.py $LAB $FE prod/$n --side-csv $SIDE $ar >> logs/full_$n.log 2>&1
    echo "$(date "+%m-%d %H:%M") $(hostname | cut -c1-8) full $n rc=$? $(grep PROBE logs/full_$n.log | tail -1 | cut -c1-200)" >> logs/push90.log
    [ -f prod/$n/summary.json ] && { ok=1; break; }
  done
  [ $ok = 1 ] || { echo "$(date "+%m-%d %H:%M") full $n FAILED twice -> lane stops" >> logs/push90.log; exit 1; }
done
