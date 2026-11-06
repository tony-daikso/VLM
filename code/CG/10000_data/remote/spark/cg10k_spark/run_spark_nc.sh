#!/bin/bash
# Spark helper for the CT-rate cg10k OWL 3 mm extraction of set `nc` (non-contrast series, 2026-11-06). Resumable.
#   waits until the Spark v15 run (run_spark15.sh / vert15_full.py) is gone (3 checks), then
#   spark_coord_nc.py claims nc cases on CT-rate (desc order; CT-rate works asc), pulls, syncs back, releases;
#   N_W workers of the unchanged src/cg10k_run.py on in/nc -> out/nc.
# usage: tmux new -d -s cg10k_nc_spark "bash ~/OWL/cg10k_spark/run_spark_nc.sh 2>&1 | tee -a ~/OWL/cg10k_spark/logs/run_spark_nc.log"
# stop:  touch ~/OWL/cg10k_spark/STOP
N_W=${N_W:-2}; THREADS=${THREADS:-6}; RECYCLE=${RECYCLE:-300}
D=$HOME/OWL/cg10k_spark; PY=$HOME/radar_env/bin/python
cd $D; mkdir -p logs state in/nc out
ts() { date "+%F %T"; }
[ -f STOP ] && { echo "$(ts) STOP exists - remove it to run"; exit 1; }
n=0; last=""
while [ $n -lt 3 ]; do
  if pgrep -f run_spark15.sh >/dev/null || pgrep -f vert15_full.py >/dev/null; then
    n=0; h=$(date +%H); [ "$h" != "$last" ] && { last=$h; echo "$(ts) waiting for Spark v15 run"; }
  else n=$((n+1)); fi
  sleep 60
done
echo "$(ts) Spark v15 gone -> start nc"
rm -f in/nc/.SYNC_DONE state/STOP state/STOP_*
rm -f in/nc/.*.nii.gz.* 2>/dev/null
$PY -u spark_coord_nc.py >> logs/coord_nc.log 2>&1 < /dev/null &
cpid=$!; echo "$(ts) coord pid $cpid"
for i in $(seq 0 $((N_W-1))); do
  ( while :; do
      OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u src/cg10k_run.py --set nc=$D/in/nc=$D/out/nc \
        --state $D/state --device cuda --threads $THREADS --tag spnc$i --watch --stable-sec 10 --idle-exit-h 24 \
        --max-cases $RECYCLE >> logs/spnc$i.log 2>&1 < /dev/null
      rc=$?; [ $rc -eq 10 ] || { echo "$(ts) spnc$i exit rc=$rc"; break; }
      [ -f STOP ] && break
    done ) &
  echo "$(ts) started spnc$i loop pid $!"; sleep 10
done
wait $cpid; echo "$(ts) coord exited rc=$?"
[ -f STOP ] && touch state/STOP
wait
echo "$(ts) ALL DONE (synced: $(grep -c synced logs/coord_nc.log) batches)"
