#!/bin/bash
# A6000: pull CT-rate helper crops (crops_h) into crops/ every 90 s until the A6000 crop job has finished.  usage: ru_assist_a6.sh UNIT CODE
U=$1; C=$2; cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
while pgrep -f "[r]u_crop.py $C ids_a6.txt" > /dev/null; do
  rsync -a --exclude '*.tmp*' daikso@192.168.0.189:OWL/mw_exp/region_units_tmp/$U/crops_h/ $U/crops/ 2>/dev/null
  sleep 90
done
echo ASSIST_DONE $(date +%T)
