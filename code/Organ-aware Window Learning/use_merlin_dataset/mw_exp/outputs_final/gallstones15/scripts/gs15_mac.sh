#!/bin/bash
# Mac side of gs15: NW crop workers reading the original CT from the Passport + a sync loop with A6000 (every 2 min):
#   push finished crops (removed locally after transfer), push/pull claims (add-only), push logs, pull STOP.
# Usage: nohup caffeinate -i bash gs15_mac.sh NW > gs15_stage/mac_driver.log 2>&1 &
# Stop: touch gs15_stage/root/STOP (or A6000 outputs_final/gallstones15/STOP -> propagated).
ST=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/gs15_stage; RT=$ST/root; PY=/Users/tony.tu/miniconda3/bin/python; NW=${1:-4}
CTD="/Volumes/My Passport/Tony/VLM/data/CT/merlin/merlin_data"; A=A6000:Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones15
mkdir -p $RT/crops $RT/claims $RT/logs
for i in $(seq 0 $((NW-1))); do
  nohup bash -c "while [ ! -f $RT/STOP ] && [ ! -f $RT/CROP_DONE_mac_w$i ]; do nice -n 10 $PY $ST/scripts/gs15_crop.py --ids $ST/ids_mac.txt --maskmap $ST/maskmap_mac.tsv --gs6 $ST/gs6 --root $RT --ct-dir '$CTD' --tag mac --wid $i --max-cases 300 --threads 2; sleep 2; done" > $RT/logs/loop_w$i.log 2>&1 < /dev/null &
  echo "mac crop worker $i pid $!"; sleep 1
done
while true; do
  ssh -o BatchMode=yes A6000 "test -f ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones15/STOP" && touch $RT/STOP
  rsync -a --remove-source-files --include='AC*.npz' --exclude='*' $RT/crops/ $A/crops/ 2>>$ST/sync_err.log
  rsync -a $RT/claims/ $A/claims/ 2>>$ST/sync_err.log
  rsync -a $A/claims/ $RT/claims/ 2>>$ST/sync_err.log
  ssh -o BatchMode=yes A6000 "mkdir -p ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones15/logs/mac"
  rsync -a $RT/logs/crop_info_mac_w*.jsonl $A/logs/mac/ 2>>$ST/sync_err.log; [ -f $RT/failed_mac.txt ] && rsync -a $RT/failed_mac.txt $A/logs/mac/
  n=$(ls $RT/claims | wc -l); echo "$(date +%H:%M) sync ok claims $n local_crops $(ls $RT/crops | wc -l)"
  if [ -f $RT/STOP ] || [ $(ls $RT/CROP_DONE_mac_w* 2>/dev/null | wc -l) -ge $NW ]; then
    if ! pgrep -f "gs15_crop.py" > /dev/null; then
      rsync -a --remove-source-files --include='AC*.npz' --exclude='*' $RT/crops/ $A/crops/; rsync -a $RT/claims/ $A/claims/
      rsync -a $RT/logs/crop_info_mac_w*.jsonl $A/logs/mac/; echo "MAC_DONE $(date)"; break
    fi
  fi
  sleep 120
done
