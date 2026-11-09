#!/bin/bash
# rib_straight tile-probe production (agent rib_straight): rib_probe3 (rib_v3 recipe) on straightened-tile frozen features,
# all train 5-fold OOF (t_rib, sitewise folds) + val fold-mean; 5 seeds per run. usage: rs_prod.sh HOST FEATS NSEC "name:seed0:side:lev" ...
# each run at most 2 tries; a run failing twice stops the lane. stop: touch <base>/STOP_PROD (between runs)
H=$1; FE=$2; NS=$3; shift 3
case $H in
 a6) B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_straight; V=$B/../rib_v3; PY=~/miniconda3/envs/radar/bin/python;;
 sp) B=~/Desktop/Tony/OWL/rib_straight; V=~/Desktop/Tony/OWL/rib_v3; PY=~/radar_env/bin/python;;
esac
cd $B; mkdir -p prod logs
for spec in "$@"; do
  IFS=: read n s0 sd lv <<< "$spec"; ok=0
  for try in 1 2; do
    [ -f prod/$n/summary.json ] && { ok=1; break; }; [ -f STOP_PROD ] && exit 0
    echo "$(date '+%m-%d %H:%M') $H $n start try $try" >> logs/prod.log
    OMP_NUM_THREADS=3 nice -n 5 $PY $V/scripts/rib_probe3.py $V/labels.csv $FE prod/$n --side-csv $V/rib_side.csv --seeds 5 --seed0 $s0 \
      --side $sd --lev $lv --nsec $NS >> logs/prod_$n.log 2>&1
    echo "$(date '+%m-%d %H:%M') $H $n rc=$? $(grep PROBE logs/prod_$n.log | tail -1 | cut -c1-200)" >> logs/prod.log
    [ -f prod/$n/summary.json ] && { ok=1; break; }
  done
  [ $ok = 1 ] || { echo "$(date '+%m-%d %H:%M') $H $n FAILED twice -> lane stops" >> logs/prod.log; exit 1; }
done
