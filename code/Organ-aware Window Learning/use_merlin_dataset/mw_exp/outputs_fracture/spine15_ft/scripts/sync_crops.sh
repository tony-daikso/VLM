#!/bin/bash
# pull spine15 crops CT-rate -> A6000 for the ids in $1 (file list), every 2 min, until all present. stop: touch /home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/spine15_ft/STOP_SYNC
S=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/spine15_ft; L=$1; R=daikso@192.168.0.189:/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/crops/
while [ ! -f $S/STOP_SYNC ]; do
  rsync -a --ignore-missing-args --files-from=$L $R $S/crops/ 2>> $S/logs/sync.err
  n=$(comm -12 <(ls $S/crops | sort) <(sort $L) | wc -l); echo "$(date +%H:%M) $n/$(wc -l < $L)"
  [ $n -ge $(wc -l < $L) ] && { echo SYNC_DONE; break; }
  sleep 120
done
