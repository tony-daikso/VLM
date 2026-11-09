#!/bin/bash
# push90 A: tier probes on v2 feats (stages 0-3) once every tier id is in A/feats_v2. usage: a_tier_v2.sh "name|args" ...
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90; RS=../rib_straight; PY=~/miniconda3/envs/radar/bin/python
sort $RS/in/ids_tier.txt > .tier_ids
until [ $(ls A/feats_v2 | grep -v tmp | sed "s/.npz//" | sort | comm -12 - .tier_ids | wc -l) -ge 2005 ]; do [ -f STOP ] && exit 0; sleep 60; done
mkdir -p A/tier_v2; for f in $(cat .tier_ids); do ln -sf ../feats_v2/$f.npz A/tier_v2/$f.npz; done
for spec in "$@"; do
  n=${spec%%|*}; ar=${spec#*|}
  for try in 1 2; do
    [ -f A/runs/$n/summary.json ] && break; [ -f STOP ] && exit 0
    echo "$(date "+%m-%d %H:%M") a6 tier $n start try $try" >> logs/push90.log
    OMP_NUM_THREADS=2 nice -n 5 $PY scripts/rs_probe2.py $RS/in/labels_tier_probe.csv A/tier_v2 A/runs/$n --side-csv $RS/in/rib_side.csv --seeds 3 --side 0.5 --lev 0.3 $ar >> logs/a_$n.log 2>&1
    echo "$(date "+%m-%d %H:%M") a6 tier $n rc=$? $(grep PROBE logs/a_$n.log | tail -1 | cut -c1-200)" >> logs/push90.log
  done
done
