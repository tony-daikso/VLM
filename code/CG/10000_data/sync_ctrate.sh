#!/bin/bash
# Periodically rsync converted NIfTI + records + series_selection.csv to CT-rate; final sync after converter exits.
# Writes run_status_20261105/final_sync_ok only when the converter exited 0 AND the final rsync succeeded.
# usage: nohup ./sync_ctrate.sh [pid-to-wait-for-first] > sync.log 2>&1 &
SRC=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/nifti_3mm/
DST=CT-rate:/media/daikso/CTDATA/cg10k/nifti_3mm/
ST=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/run_status_20261105
RS="rsync -a --partial --exclude *.tmp* --exclude records_bad_* --exclude convert_run0.log"
rm -f $ST/final_sync_ok
if [ -n "$1" ]; then echo "$(date '+%F %T') waiting for pid $1 (previous sync) to exit"; while kill -0 $1 2>/dev/null; do sleep 30; done; fi
ssh CT-rate 'mkdir -p /media/daikso/CTDATA/cg10k/nifti_3mm'
while true; do
  alive=0; pgrep -f "select_convert.py --workers" >/dev/null && alive=1
  echo "$(date '+%F %T') sync start (converter alive=$alive)"
  if [ $alive -eq 0 ]; then
    for i in 1 2 3 4 5; do
      $RS "$SRC" "$DST" && { ok=1; break; }; ok=0; echo "final rsync failed (try $i)"; sleep 120
    done
    echo "$(date '+%F %T') final sync ok=$ok: $(ls "$SRC" | grep -c 'nii.gz$') local nii; converter exit=$(cat $ST/main_exit_code 2>/dev/null)"
    [ $ok -eq 1 ] && [ "$(cat $ST/main_exit_code 2>/dev/null)" = "0" ] && touch $ST/final_sync_ok
    echo "converter finished; final sync done"; break
  fi
  $RS "$SRC" "$DST" && echo "$(date '+%F %T') sync ok: $(ls "$SRC" | grep -c 'nii.gz$') local nii"
  sleep 600
done
