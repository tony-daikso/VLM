#!/bin/bash
# A6000 gs15 crop workers + GPU-extract keepalive (15:30 restart; the 12:12 loops were children of run_gs15.sh and died with it at
# 12:41, workers finished their 300-case batches by 13:23). Disk guard: no new batch while / has < 15 GB free (the crop script
# also stops its batch below 15 GB); ct/ temp files older than 30 min are removed. Stop: touch STOP (all) or STOP_a6 (crops only).
MW=$HOME/Desktop/Tony/VLM/OWL/mw_exp; R=$MW/outputs_final/gallstones15; S=$R/scripts; PY=$HOME/miniconda3/envs/radar/bin/python; NW=${1:-7}
TS=$HOME/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
G6=$MW/outputs_final/region_units/gallstones/gs6_seg; cd $R
free_gb () { df -BG --output=avail $R | tail -1 | tr -dc 0-9; }
for i in $(seq 0 $((NW-1))); do
  bash -c "while [ ! -f $R/STOP ] && [ ! -f $R/STOP_a6 ] && [ ! -f $R/CROP_DONE_a6_w$i ]; do
    if [ \$(df -BG --output=avail $R | tail -1 | tr -dc 0-9) -lt 15 ]; then echo \"\$(date +%H:%M) disk guard: waiting\"; sleep 120; continue; fi
    nice -n 10 $PY $S/gs15_crop.py --ids $R/inputs/ids_a6.txt --maskmap $R/inputs/maskmap.tsv --gs6 $G6 --root $R --ct-ssh --tag a6 --wid $i --max-cases 50 --threads 2; sleep 2; done" >> logs/crop_a6_w$i.log 2>&1 < /dev/null &
  echo "crop worker $i loop pid $!"; sleep 1
done
echo "WORKERS_START $(date) pid $$"
while [ ! -f $R/STOP ]; do
  find $R/ct -type f -mmin +30 -delete 2>/dev/null
  if ! pgrep -f "gs15_extract.py --crops $R/crops" > /dev/null && [ ! -f $R/EXTRACT_DONE ]; then
    echo "$(date +%H:%M) extract not running -> restart"
    (cd $MW && MW_ROOT=. $PY $S/gs15_extract.py --crops $R/crops --out $R/emb --ts-model-dir $TS --windows $S/gs15_windows.yaml --poll --done-flag $R/CROP_ALL_DONE --device cuda --vram-gb 12 --threads 2 >> $R/logs/extract.log 2>&1 && [ -f $R/CROP_ALL_DONE ] && touch $R/EXTRACT_DONE) &
  fi
  echo "$(date +%H:%M) free $(free_gb) GB crops $(ls $R/crops | wc -l) emb $(ls $R/emb | wc -l)"
  sleep 120
done
echo "WORKERS_EXIT $(date)"
