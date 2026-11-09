#!/bin/bash
# pelvis15 full feature extraction on the CT-rate 3080 for encoder $1 (polls new ROIs until crops are done).
# usage: run_feat_full.sh 292|294|295   stop: touch .../pelvis15/f_$1/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15; PY=~/miniconda3/envs/radar/bin/python; e=$1
case $e in 294) T=$(ls -d /media/daikso/Tony/VLM/models/ts/Dataset294*/nnUNet*);; *) T=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset${e}*/nnUNet*);; esac
mkdir -p f_$e
while [ ! -f f_$e/STOP ]; do
  $PY scripts/pv15_feat.py roi f_$e --ts $T --bone --once --vram-gb 9 --ids ids_order.txt >> logs/feat_full_$e.log 2>&1
  if [ -f CROP_DRIVER_DONE ] && [ -f SPARK_DONE ]; then $PY scripts/pv15_feat.py roi f_$e --ts $T --bone --once --vram-gb 9 >> logs/feat_full_$e.log 2>&1; break; fi
  sleep 120
done
touch f_$e/FULL_DONE
