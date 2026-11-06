#!/bin/bash
# Wait until NOTHING else reads the USB drive, then build the header-only DICOM index.
# usage: nohup ./wait_then_index.sh <pid> [<pid>...] > .../dicom_index/wait_index.log 2>&1 &
cd /Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data
OUT=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/dicom_index
VOL=/Volumes/PolypXfer
WAIT_PIDS="$*"   # run_main_resume.sh, run_delay_after_main.sh, sync_ctrate.sh (2026-11-05 resume)
ST=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/run_status_20261105
busy() {
  for p in $WAIT_PIDS; do kill -0 $p 2>/dev/null && { echo "pid $p alive"; return 0; }; done
  pgrep -f "select_convert.py|convert_delay.py|run_delay_after_main.sh|run_main_resume.sh|sync_ctrate.sh" >/dev/null && { echo "converter process running"; return 0; }
  pgrep -f "rsync.*PolypXfer" >/dev/null && { echo "rsync on $VOL running"; return 0; }
  return 1
}
echo "$(date '+%F %T') waiting (pids $WAIT_PIDS, converters, rsync on $VOL); check every 60 s"
n=0
while true; do
  if why=$(busy); then n=0; else n=$((n+1)); fi
  [ $n -ge 2 ] && break          # must be idle on 2 consecutive checks (gap between chain steps)
  [ $(( $(date +%s) / 60 % 30 )) -eq 0 ] && echo "$(date '+%F %T') still waiting: $why"
  sleep 60
done
echo "$(date '+%F %T') drive idle; open files on $VOL (info only):"
lsof -Fpc +f -- $VOL 2>/dev/null | paste - - | head -20
if [ ! -d $VOL/deliver_10k/images/2022 ]; then echo "$(date '+%F %T') $VOL not mounted -> abort"; exit 1; fi
if [ "$(cat $ST/main_exit_code 2>/dev/null)" != "0" ] || [ ! -f $ST/final_sync_ok ] || [ ! -f $ST/delay_chain_ok ]; then
  echo "$(date '+%F %T') main exit=$(cat $ST/main_exit_code 2>/dev/null) final_sync_ok=$([ -f $ST/final_sync_ok ] && echo y || echo n) delay_chain_ok=$([ -f $ST/delay_chain_ok ] && echo y || echo n) -> abort (drive disconnect or failed step)"; exit 1
fi
T0=$(date +%s)
caffeinate -i python -u build_dicom_index.py --workers 2 --smoke 3 --bench 7 --bench-workers 1,2,3 > $OUT/build_index.log 2>&1
rc=$?
echo "$(date '+%F %T') build_dicom_index exit=$rc wall=$(( $(date +%s) - T0 ))s"
tail -3 $OUT/build_index.log
