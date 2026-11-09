#!/bin/bash
# pelvis15 crop driver (CT-rate CPU): 4 procs while the spine15 crop still runs, 12 once it is gone.
# stop: touch /media/daikso/Tony/VLM/data/CT/merlin/pelvis15/STOP_ALL
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15
PY=~/miniconda3/envs/radar/bin/python; M="$HOME/OWL/mw_exp/outputs/masks ../frac15_raw/masks3"
while [ ! -f STOP_ALL ]; do
  if pgrep -f spine15_crop.py >/dev/null; then N=4; else N=12; fi
  rm -f STOP; echo "$(date '+%m-%d %H:%M') start N=$N" >> logs/driver.log
  nice -n 19 ionice -c2 -n7 $PY scripts/pv15_crop.py ids_order.txt . $N $M >> logs/crop.log 2>&1 &
  P=$!; up=0
  while kill -0 $P 2>/dev/null; do
    sleep 60
    if [ -f STOP_ALL ]; then touch STOP; fi
    if [ $N = 4 ] && ! pgrep -f spine15_crop.py >/dev/null; then touch STOP; up=1; fi
  done
  wait $P
  [ $up = 1 ] && continue
  break
done
echo "$(date '+%m-%d %H:%M') driver end" >> logs/driver.log; touch CROP_DRIVER_DONE
