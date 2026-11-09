#!/bin/bash
# push90 B: PE tier probes on the 3080 (rs_probe2, t_pe, nsec 4, nbmode tile, 3 seeds). Waits until every tier id has a crop
# status and every ok crop has feats. usage: pe_tier.sh 'name|args' ... ; 2 tries max; stop: touch /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/STOP_PROBE
cd /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15; mkdir -p runs; PY=~/miniconda3/envs/radar/bin/python
cut -d, -f1 pe_probe_labels_tier.csv | tail -n +2 | sort > .tier_ids
while true; do
  [ -f STOP_PROBE ] && exit 0
  nst=$(cut -f1 status.tsv | sort -u | comm -12 - .tier_ids | wc -l)
  nok=$(grep -P '\tok' status.tsv | cut -f1 | sort -u | comm -12 - .tier_ids | wc -l)
  nf=$(ls feats | sed 's/.npz//' | sort | comm -12 - .tier_ids | wc -l)
  [ $nst -ge $(wc -l < .tier_ids) ] && [ $nf -ge $nok ] && break
  sleep 60
done
for spec in "$@"; do
  n=${spec%%|*}; ar=${spec#*|}
  for try in 1 2; do
    [ -f runs/$n/summary.json ] && break; [ -f STOP_PROBE ] && exit 0
    echo "$(date '+%m-%d %H:%M') ct pe tier $n start try $try" >> logs/push90.log
    OMP_NUM_THREADS=2 nice -n 5 $PY scripts/rs_probe2.py pe_probe_labels_tier.csv feats runs/$n --side-csv pe_side.csv --seeds 3 --nsec 4 --nbmode tile --target t_pe $ar >> logs/pe_$n.log 2>&1
    echo "$(date '+%m-%d %H:%M') ct pe tier $n rc=$? $(grep PROBE logs/pe_$n.log | tail -1 | cut -c1-200)" >> logs/push90.log
  done
done
