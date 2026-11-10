#!/bin/bash
# delete crops once their embeddings are DONE (ath: also after its HU scalars exist); log free disk space
cd ~/Desktop/Tony/VLM/OWL/cg10k_best
while true; do
  for u in hn fa asc ath rlq bd5 frp gs5 gs; do for s in main delay; do
    [ -d bcrops/$u/$s ] || continue; [ -f bemb/$u/$s.DONE ] || continue
    [ $u = ath ] && [ ! -s bemb/ath_hufeat_$s.csv ] && continue
    rm -rf bcrops/$u/$s && echo "$(date +%T) removed crops $u/$s"
  done; done
  echo "$(date +%T) free $(df -h ~ | tail -1 | awk "{print \$4}")"
  sleep 300
done
