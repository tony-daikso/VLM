#!/bin/bash
# Spark helper for the CT-rate cg10k OWL 3 mm extraction (2026-11-06). Resumable.
#   spark_coord.py: claims cases on CT-rate (delay asc, then main from the tail), pulls NIfTIs, syncs results back,
#                   releases claims; at the end re-runs infer_head on CT-rate if needed.
#   N_W workers of the unchanged src/cg10k_run.py on in/<set> -> out/<set>.
# usage: tmux new -d -s cg10k_spark 'bash ~/OWL/cg10k_spark/run_spark.sh 2>&1 | tee -a ~/OWL/cg10k_spark/logs/run_spark.log'
# stop:  touch ~/OWL/cg10k_spark/STOP  (no new claims; in-flight cases finish + sync; claims released; workers exit)
N_W=${N_W:-2}; THREADS=${THREADS:-6}; RECYCLE=${RECYCLE:-300}
D=$HOME/OWL/cg10k_spark; PY=$HOME/radar_env/bin/python
cd $D; mkdir -p logs state in/main in/delay out
ts() { date '+%F %T'; }
[ -f STOP ] && { echo "$(ts) STOP exists - remove it to run"; exit 1; }
rm -f in/*/.SYNC_DONE state/STOP state/STOP_*
rm -f in/*/.*.nii.gz.* 2>/dev/null   # partial rsync temp files

$PY -u spark_coord.py >> logs/coord.log 2>&1 < /dev/null &
cpid=$!; echo "$(ts) coord pid $cpid"
pids=()
for i in $(seq 0 $((N_W-1))); do
  ( while :; do
      OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u src/cg10k_run.py --set delay=$D/in/delay=$D/out/delay --set main=$D/in/main=$D/out/main \
        --state $D/state --device cuda --threads $THREADS --tag sp$i --watch --stable-sec 10 --idle-exit-h 24 \
        --max-cases $RECYCLE >> logs/sp$i.log 2>&1 < /dev/null
      rc=$?; [ $rc -eq 10 ] || { echo "$(ts) sp$i exit rc=$rc"; break; }
      [ -f STOP ] && break
    done ) &
  pids+=($!); echo "$(ts) started sp$i loop pid $!"; sleep 10
done
wait $cpid; echo "$(ts) coord exited rc=$?"
[ -f STOP ] && touch state/STOP   # workers exit after their current case
wait
echo "$(ts) ALL DONE (synced: $(grep -c synced logs/coord.log) batches)"
