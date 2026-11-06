#!/bin/bash
# C- (non-contrast) pass chain, 2026-11-06:
#   1) wait until build_dicom_index.py (and wait_then_index.sh) has exited, files.parquet + series.csv exist, the
#      final 'merged' line covers >= 9900 exams, and the drive is mounted. Never reads the drive while the index runs.
#   2) smoke: 3 exams whose MAIN series is C- re-converted through the index path must be voxel-identical to nifti_3mm
#   3) convert_nc.py (caffeinate -i, 2 workers) -> nifti_3mm_nc/; rsync to CT-rate every 10 min while running
#   4) final rsync; only if convert exit 0 + rsync ok + drive still mounted -> CT-rate nifti_3mm_nc/.SYNC_DONE
# usage: nohup caffeinate -i ./run_nc_chain.sh > .../nifti_3mm_nc/chain.log 2>&1 &
cd /Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data
PY=/Users/tony.tu/miniconda3/envs/VLM/bin/python
BASE=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data
IDX=$BASE/dicom_index; OUT=$BASE/nifti_3mm_nc
DRV=/Volumes/PolypXfer/deliver_10k/images/2022
DST=CT-rate:/media/daikso/CTDATA/cg10k/nifti_3mm_nc/
RS="rsync -a --partial --exclude *.tmp* --exclude _smoke --exclude chain.log --exclude convert_nc.log"
ts() { date '+%F %T'; }
mkdir -p $OUT
echo "$(ts) waiting for build_dicom_index.py / wait_then_index.sh to exit"
n=0
while true; do
  if pgrep -f "build_dicom_index.py|wait_then_index.sh|select_convert.py|convert_delay.py" >/dev/null; then n=0; else n=$((n+1)); fi
  [ $n -ge 2 ] && break
  sleep 60
done
last=$(grep '^merged ' $IDX/build_index.log | tail -1)
nm=$(echo "$last" | awk '{print $2}')
echo "$(ts) index process gone; last merge: $last"
[ -f $IDX/files.parquet ] && [ -f $IDX/series.csv ] || { echo "files.parquet/series.csv missing -> abort"; exit 1; }
[ "${nm:-0}" -ge 9900 ] || { echo "final merge covers only ${nm:-0} exams (index incomplete / drive disconnect?) -> abort"; exit 1; }
[ -d $DRV ] || { echo "$DRV not mounted -> abort"; exit 1; }

echo "$(ts) smoke compare"
$PY -u convert_nc.py --smoke-compare 3 || { echo "$(ts) smoke failed (exit $?) -> abort"; exit 1; }

ssh CT-rate "mkdir -p /media/daikso/CTDATA/cg10k/nifti_3mm_nc"
$PY -u convert_nc.py --workers 2 > $OUT/convert_nc.log 2>&1 &
cpid=$!
echo "$(ts) convert_nc.py pid $cpid (log $OUT/convert_nc.log)"
while kill -0 $cpid 2>/dev/null; do
  for i in $(seq 60); do kill -0 $cpid 2>/dev/null || break; sleep 10; done
  $RS $OUT/ $DST && echo "$(ts) sync ok: $(ls $OUT | grep -c 'nii.gz$') local nii"
done
wait $cpid; rc=$?
echo "$(ts) convert_nc exit=$rc"; tail -2 $OUT/convert_nc.log
ok=0
for i in 1 2 3 4 5; do $RS $OUT/ $DST && { ok=1; break; }; echo "final rsync failed (try $i)"; sleep 120; done
echo "$(ts) final rsync ok=$ok: local $(ls $OUT | grep -c 'nii.gz$') nii, remote $(ssh CT-rate "ls /media/daikso/CTDATA/cg10k/nifti_3mm_nc | grep -c 'nii.gz\$'")"
if [ $rc -eq 0 ] && [ $ok -eq 1 ] && [ -d $DRV ]; then
  until ssh -o ConnectTimeout=20 CT-rate "touch /media/daikso/CTDATA/cg10k/nifti_3mm_nc/.SYNC_DONE"; do sleep 120; done
  echo "$(ts) .SYNC_DONE written"
else
  echo "$(ts) NOT writing .SYNC_DONE (rc=$rc rsync_ok=$ok drive=$([ -d $DRV ] && echo y || echo n)); rerun this script to resume"
  exit 1
fi
