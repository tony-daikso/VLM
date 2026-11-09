#!/bin/bash
# RLQ crop-unit chain on A6000: (loop) rsync crops from CT-rate -> GPU extract (skips done) until CT-rate CROP_DONE,
# then GPU probe. Holds outputs_final/A6000_GPU_LOCK while running; removes it at exit.
# Run from anywhere:  setsid nohup bash outputs_final/appendicitis_rlq_unit/scripts/run_rlq.sh > outputs_final/appendicitis_rlq_unit/run_rlq.log 2>&1 &
cd ~/Desktop/Tony/VLM/OWL/mw_exp
U=outputs_final/appendicitis_rlq_unit; PY=~/miniconda3/envs/radar/bin/python; CT=daikso@192.168.0.189; CTD=OWL/mw_exp/analysis_appendicitis_rlq_tmp
LOCK=outputs_final/A6000_GPU_LOCK
if [ -f $LOCK ]; then echo "lock held: $(cat $LOCK)"; exit 1; fi
echo "appendicitis_rlq_unit (rlq extract+probe) pid $$ since $(date '+%F %T')" > $LOCK
trap "rm -f $PWD/$LOCK" EXIT
mkdir -p $U/crops $U/emb_rlq
while true; do
  done_flag=$(ssh $CT "grep -c CROP_DONE $CTD/crop_all.log" 2>/dev/null)
  rsync -a --exclude '*.tmp*' $CT:$CTD/crops/ $U/crops/
  MW_ROOT=. nice -n 10 $PY $U/scripts/rlq_extract.py --crops $U/crops --out $U/emb_rlq --ids $U/ids_trval.txt --device cuda --config config_final.yaml >> $U/extract.log 2>&1
  echo "$(date '+%T') emb $(ls $U/emb_rlq | wc -l) crops $(ls $U/crops | wc -l) done_flag=$done_flag"
  [ "$done_flag" = "1" ] && break
  sleep 120
done
rsync -a --exclude '*.tmp*' $CT:$CTD/crops/ $U/crops/
MW_ROOT=. nice -n 10 $PY $U/scripts/rlq_extract.py --crops $U/crops --out $U/emb_rlq --ids $U/ids_trval.txt --device cuda --config config_final.yaml >> $U/extract.log 2>&1
echo "EXTRACT_FINAL emb $(ls $U/emb_rlq | wc -l)"
NT=4 nice -n 10 $PY $U/scripts/rlq_probe.py $U/emb_rlq $U/out --device cuda > $U/probe.log 2>&1
echo RLQ_CHAIN_DONE
