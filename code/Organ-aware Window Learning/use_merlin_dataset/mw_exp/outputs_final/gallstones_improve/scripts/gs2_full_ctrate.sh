#!/bin/bash
# GS2 full run plan (NOT launched). Run AFTER Exp J has finished (~22:30, 2026-11-05) so the CT-rate HDD is free.
# Native-resolution GB features for all 20,364 train+val cases. Test is NOT included (only after the config is locked on val).
set -e
# --- 0) from the Mac: confirm Exp J is finished (all three must print nothing) -------------------------------
#   ssh sparks  'pgrep -af "vert15_full|expJ" | grep -v pgrep'
#   ssh A6000   'pgrep -af "vert15_full|expJ" | grep -v pgrep'
#   ssh CT-rate 'pgrep -af "vert15_full|expJ|extract" | grep -v pgrep; cat /proc/loadavg'
# --- 1) from the Mac: push scripts + id list to CT-rate -------------------------------------------------------
#   ssh CT-rate 'mkdir -p ~/OWL/gs_improve/{scripts,in,out}'
#   scp sparks:OWL/gs_improve/scripts/{gs2_native.py,gs1_stack.py} CT-rate:OWL/gs_improve/scripts/   (via Mac)
#   scp sparks:OWL/gs_improve/in/trval.txt CT-rate:OWL/gs_improve/in/
# --- 2) prep npz (3 mm GB mask) : CT-rate SSD has 12,792 / 20,364 -> copy the other 7,572 (~23 GB) from Spark ----
#   ssh CT-rate 'ls ~/OWL/mw_exp/outputs/prep | sed s/.npz// | sort > /tmp/ct_prep.txt; comm -23 ~/OWL/gs_improve/in/trval.txt /tmp/ct_prep.txt | sed s/$/.npz/ > ~/OWL/gs_improve/in/prep_missing.txt; wc -l ~/OWL/gs_improve/in/prep_missing.txt'
#   scp CT-rate:OWL/gs_improve/in/prep_missing.txt sparks:/tmp/gs_prep_missing.txt
#   ssh sparks 'rsync -a --files-from=/tmp/gs_prep_missing.txt ~/OWL/mw_exp/outputs/prep/ daikso@192.168.0.189:OWL/mw_exp/outputs/prep/'
# --- 3) on CT-rate: the run (6 processes, nice/ionice; resumable: re-running skips finished ids) ---------------
cd ~/OWL/gs_improve
nohup nice -n 10 ionice -c2 -n7 ~/miniconda3/envs/radar/bin/python scripts/gs2_native.py \
  in/trval.txt /media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data ~/OWL/mw_exp/outputs/prep \
  out/gs2_native_trval.csv 6 > out/gs2_native_trval.log 2>&1 &
echo $! > out/gs2_native_trval.pid
# progress: tail -f out/gs2_native_trval.log  (prints every 25 cases);  expected ~5-6 h (HDD ~2.37 TB read)
# --- 4) afterwards (Spark, CPU, ~15 min): stack with native features, selection on train OOF only ---------------
#   scp CT-rate:OWL/gs_improve/out/gs2_native_trval.csv sparks:OWL/gs_improve/out/   (via Mac)
#   ssh sparks 'cd ~/OWL/gs_improve; OMP_NUM_THREADS=4 nice -n 19 ~/radar_env/bin/python scripts/gs1_stack.py in out/gs1_feat3mm.csv out/gs1_nat out/gs2_native_trval.csv > out/gs1_nat.log 2>&1'
#   go / no-go: keep V5a/V5b only if train OOF delta vs V4 >= +0.005 with a bootstrap CI above 0; val is reported once.
