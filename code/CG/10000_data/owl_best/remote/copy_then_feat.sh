#!/bin/bash
# copy 3 mm prep+masks once from the HDD to local SSD (idle IO class), then run features from SSD
for s in main delay; do for d in prep masks; do
  ionice -c3 nice -n 19 rsync -a /media/daikso/CTDATA/cg10k/owl/$s/$d/ ~/OWL/cg10k_best/data/$s/$d/
done; echo "$(date) copied $s"; done
~/OWL/cg10k_best/run_feat.sh main 8 > ~/OWL/cg10k_best/logs/feat_main.log 2>&1
~/OWL/cg10k_best/run_feat.sh delay 8 > ~/OWL/cg10k_best/logs/feat_delay.log 2>&1
