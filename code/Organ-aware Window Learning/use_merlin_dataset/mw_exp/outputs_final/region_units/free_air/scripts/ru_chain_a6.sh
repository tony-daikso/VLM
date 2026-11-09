#!/bin/bash
# A6000 CPU chain for Exp M crops: after the hn crop job, crop fa then asc (A6000 share, 4 proc, nice 19) with the
# crops_h pull loop from the CT-rate helper.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
PY=~/miniconda3/envs/radar/bin/python
while pgrep -f "[r]u_crop.py hn ids_a6.txt" > /dev/null; do sleep 30; done
for pair in fa:free_air asc:ascites; do C=${pair%%:*}; U=${pair##*:}
  nice -n 19 $PY scripts/ru_crop.py $C ids_a6.txt $U/crops 4 ../../outputs/prep:../../outputs/masks > $U/crop_a6.log 2>&1 &
  sleep 5; bash scripts/ru_assist_a6.sh $U $C > $U/assist.log 2>&1
  wait
done
echo CHAIN_A6_DONE $(date +%T)
