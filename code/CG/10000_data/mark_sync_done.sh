#!/bin/bash
# Touch .SYNC_DONE sentinels on CT-rate once the Mac-side transfers have finished, so the CT-rate OWL workers
# (~/OWL/cg10k_infer/src/cg10k_run.py --watch) know a set is complete.
#   main : after sync_ctrate.sh (pid $1) exits, only if run_status_20261105/final_sync_ok exists
#          (= select_convert.py exit 0 + final rsync ok) and the drive is still mounted
#   delay: after run_delay_after_main.sh (pid $2) exits, only if run_status_20261105/delay_chain_ok exists
# usage: nohup ./mark_sync_done.sh <sync_pid> <delay_chain_pid> > ../../../data/CT/CG/10000_data/mark_sync_done.log 2>&1 &
R=/media/daikso/CTDATA/cg10k
ST=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/run_status_20261105
DRV=/Volumes/PolypXfer/deliver_10k/images/2022
SYNC_PID=$1; CHAIN_PID=$2
echo "$(date '+%F %T') waiting for sync pid $SYNC_PID, delay chain pid $CHAIN_PID"
while kill -0 $SYNC_PID 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') sync_ctrate.sh exited"
if [ ! -f $ST/final_sync_ok ] || [ ! -d $DRV ]; then
  echo "$(date '+%F %T') final_sync_ok missing or drive gone (main exit=$(cat $ST/main_exit_code 2>/dev/null)) -> NOT writing .SYNC_DONE; abort"; exit 1
fi
until ssh -o ConnectTimeout=20 CT-rate "touch $R/nifti_3mm/.SYNC_DONE"; do sleep 120; done
echo "$(date '+%F %T') main .SYNC_DONE written ($(ssh CT-rate "ls $R/nifti_3mm | grep -c 'nii.gz\$'") nii on CT-rate)"
while kill -0 $CHAIN_PID 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') run_delay_after_main.sh exited"
if [ ! -f $ST/delay_chain_ok ] || [ ! -d $DRV ]; then
  echo "$(date '+%F %T') delay_chain_ok missing or drive gone -> NOT writing delay .SYNC_DONE; abort"; exit 1
fi
until ssh -o ConnectTimeout=20 CT-rate "mkdir -p $R/nifti_3mm_delay && touch $R/nifti_3mm_delay/.SYNC_DONE"; do sleep 120; done
echo "$(date '+%F %T') delay .SYNC_DONE written ($(ssh CT-rate "ls $R/nifti_3mm_delay | grep -c 'nii.gz\$'") nii on CT-rate)"
