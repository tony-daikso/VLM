#!/bin/bash
# Wait for the main select_convert.py run to exit (never read the USB drive concurrently), then
# --merge-only, the delay pass and rsync to CT-rate.  Proceeds only if the main run exited 0 and the drive is mounted.
# Writes run_status_20261105/delay_chain_ok only if delay pass exit 0 AND rsync ok.
# usage: nohup ./run_delay_after_main.sh <main-wrapper-pid> > .../nifti_3mm_delay/chain.log 2>&1 &
cd /Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data
PY=/Users/tony.tu/miniconda3/envs/VLM/bin/python
BASE=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data
OUT=$BASE/nifti_3mm_delay
ST=$BASE/run_status_20261105
DRV=/Volumes/PolypXfer/deliver_10k/images/2022
rm -f $ST/delay_chain_ok
echo "$(date '+%F %T') waiting for main converter (pid ${1:-none}) to exit"
while { [ -n "$1" ] && kill -0 $1 2>/dev/null; } || pgrep -f "select_convert.py --workers" >/dev/null; do sleep 60; done
rc=$(cat $ST/main_exit_code 2>/dev/null)
echo "$(date '+%F %T') main converter finished, exit=$rc"
[ "$rc" = "0" ] || { echo "main exit != 0 -> abort delay chain"; exit 1; }
[ -d $DRV ] || { echo "$DRV missing -> abort delay chain"; exit 1; }
$PY select_convert.py --merge-only
T0=$(date +%s)
caffeinate -i $PY -u convert_delay.py --workers 2 > $OUT/convert_delay.log 2>&1
drc=$?
echo "$(date '+%F %T') delay pass exit=$drc wall=$(( $(date +%s) - T0 ))s"
tail -2 $OUT/convert_delay.log
[ $drc -eq 0 ] || { echo "delay pass failed -> no rsync / no delay_chain_ok"; exit 1; }
ssh CT-rate 'mkdir -p /media/daikso/CTDATA/cg10k/nifti_3mm_delay'
ok=0
for i in 1 2 3 4 5; do
  rsync -a --partial --exclude '*.tmp*' --exclude chain.log --exclude 'records_bad_*' $OUT/ CT-rate:/media/daikso/CTDATA/cg10k/nifti_3mm_delay/ && { ok=1; break; }
  echo "rsync failed (try $i)"; sleep 120
done
[ $ok -eq 1 ] && echo "$(date '+%F %T') rsync ok: $(ls $OUT | grep -c 'nii.gz$') nii; remote: $(ssh CT-rate 'ls /media/daikso/CTDATA/cg10k/nifti_3mm_delay | grep -c nii.gz$')"
[ $ok -eq 1 ] && [ -d $DRV ] && touch $ST/delay_chain_ok
echo "$(date '+%F %T') chain done (delay_chain_ok=$([ -f $ST/delay_chain_ok ] && echo yes || echo no))"
