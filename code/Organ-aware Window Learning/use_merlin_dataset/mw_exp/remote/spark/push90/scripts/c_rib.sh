#!/bin/bash
# push90 C: rib tile probe (rs_probe2 == rib_probe3 recipe, side .5 lev .3, 5 seeds seed0 0, old feats) with train-noise weights.
# resumable per fold; at most 2 tries per run. stop: touch ~/Desktop/Tony/OWL/push90/STOP
cd ~/Desktop/Tony/OWL/push90; RS=~/Desktop/Tony/OWL/rib_straight; V=~/Desktop/Tony/OWL/rib_v3; PY=~/radar_env/bin/python
for spec in "$@"; do
  n=${spec%%:*}; w=${spec#*:}
  for try in 1 2; do
    [ -f runs/$n/summary.json ] && break; [ -f STOP ] && exit 0
    echo "$(date "+%m-%d %H:%M") sp $n start try $try" >> logs/push90.log
    OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rs_probe2.py $V/labels.csv $RS/feats runs/$n --side-csv $V/rib_side.csv --seeds 5 --seed0 0 --side 0.5 --lev 0.3 --wcsv C/$w >> logs/$n.log 2>&1
    echo "$(date "+%m-%d %H:%M") sp $n rc=$? $(grep PROBE logs/$n.log | tail -1 | cut -c1-220)" >> logs/push90.log
  done
done
