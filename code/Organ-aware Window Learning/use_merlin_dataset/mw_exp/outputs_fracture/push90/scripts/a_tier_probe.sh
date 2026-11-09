#!/bin/bash
# push90 A: tier probes (rs_probe2, 3 seeds, side .5 lev .3, 5 sitewise folds, tier rows 2,005) on A6000 GPU.
# usage: a_tier_probe.sh "name|args" ...  (runs sequentially in this lane; 2 tries max; stop: touch push90/STOP)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90; RS=../rib_straight; PY=~/miniconda3/envs/radar/bin/python; mkdir -p A/runs
until grep -q "A tier feats done" logs/push90.log 2>/dev/null; do sleep 30; done
for spec in "$@"; do
  n=${spec%%|*}; ar=${spec#*|}
  for try in 1 2; do
    [ -f A/runs/$n/summary.json ] && break; [ -f STOP ] && exit 0
    echo "$(date "+%m-%d %H:%M") a6 tier $n start try $try" >> logs/push90.log
    OMP_NUM_THREADS=2 nice -n 5 $PY scripts/rs_probe2.py $RS/in/labels_tier_probe.csv A/feats_tier A/runs/$n --side-csv $RS/in/rib_side.csv --seeds 3 --side 0.5 --lev 0.3 $ar >> logs/a_$n.log 2>&1
    echo "$(date "+%m-%d %H:%M") a6 tier $n rc=$? $(grep PROBE logs/a_$n.log | tail -1 | cut -c1-200)" >> logs/push90.log
  done
done
