#!/bin/bash
# CG10k OWL 3 mm extraction for the third set `nc` (non-contrast series of C+ exams, 2026-11-06). Resumable.
#   1) waits until the 1.5 mm v15 run is finished (no run_v15.sh / vert15_full.py for 3 checks in a row)
#   2) N_GPU GPU workers of src/cg10k_run.py --set nc=$R/nifti_3mm_nc=$R/owl/nc, GPU steps serialized by
#      state/gpu.lock, recycled every RECYCLE cases; --watch: they wait for files from the Mac and exit when
#      nifti_3mm_nc/.SYNC_DONE exists and nothing is left (Spark helps via claims 'spark-6736:1' in state/nc/claims)
#   3) waits until no Spark claim is left in state/nc/claims (max 4 h), then infer_head -> $R/owl/probs_nc.csv
# usage: tmux new -d -s cg10k_nc "bash ~/OWL/cg10k_infer/run_nc.sh 2>&1 | tee -a ~/OWL/cg10k_infer/logs/nc_run.log"
# stop:  touch ~/OWL/cg10k_infer/state/STOP   (workers exit after their current case)
N_GPU=${N_GPU:-3}; GPU_THREADS=${GPU_THREADS:-4}; RECYCLE=${RECYCLE:-300}; IDLE_H=${IDLE_H:-8}
D=/home/daikso/OWL/cg10k_infer; PY=/home/daikso/miniconda3/envs/radar/bin/python
R=/media/daikso/CTDATA/cg10k
SETS="--set nc=$R/nifti_3mm_nc=$R/owl/nc"
cd $D; mkdir -p logs state "$R/nifti_3mm_nc"
ts() { date '+%F %T'; }
cnt() { ls "$1" 2>/dev/null | grep -c 'npz$'; }
[ -f state/STOP ] && { echo "$(ts) state/STOP exists - remove it to run"; exit 1; }
n=0; last=""
while [ $n -lt 3 ]; do
  if pgrep -f "run_v15.sh" >/dev/null || pgrep -f "vert15_full.py" >/dev/null; then
    n=0; h=$(date +%H)
    [ "$h" != "$last" ] && { last=$h; echo "$(ts) waiting for v15 (main emb15 $(cnt $R/owl_v15/main/emb15), delay $(cnt $R/owl_v15/delay/emb15)); nc nifti so far $(ls $R/nifti_3mm_nc | grep -c 'nii.gz$')"; }
  else n=$((n+1)); fi
  sleep 60
done
echo "$(ts) v15 finished; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
rm -f state/STOP_gpu*
pids=()
for i in $(seq 0 $((N_GPU-1))); do
  ( while :; do
      OMP_NUM_THREADS=$GPU_THREADS MKL_NUM_THREADS=$GPU_THREADS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u src/cg10k_run.py $SETS --device cuda --threads $GPU_THREADS --tag nc_gpu$i \
        --watch --idle-exit-h $IDLE_H --gpu-lock $D/state/gpu.lock --max-cases $RECYCLE >> logs/nc_gpu$i.log 2>&1 < /dev/null
      rc=$?; [ $rc -eq 10 ] || { echo "$(ts) nc_gpu$i worker exit rc=$rc"; break; }
      [ -f state/STOP ] || [ -f state/STOP_nc_gpu$i ] && break
    done ) &
  pids+=($!); echo "$(ts) started nc_gpu$i loop pid $!"; sleep 10
done
wait
echo "$(ts) CT-rate workers exited; nc done $(cnt $R/owl/nc/emb_bones/bone_L400_W1800) / nifti $(ls $R/nifti_3mm_nc | grep -c 'nii.gz$')"
[ -f state/STOP ] && { echo "$(ts) stopped by state/STOP - skip inference"; exit 0; }
t_end=$(( $(date +%s) + 4*3600 ))
while [ $(date +%s) -lt $t_end ]; do
  k=$(grep -l '^spark' state/nc/claims/* 2>/dev/null | wc -l)
  [ $k -eq 0 ] && break
  echo "$(ts) waiting for $k Spark claims"; sleep 120
done
$PY -u src/infer_head.py --model $D/model/E3_late.pt --emb-dir $R/owl/nc/emb --bone-emb-dir $R/owl/nc/emb_bones \
  --per-seed --out $R/owl/probs_nc.csv > logs/infer_nc.log 2>&1
echo "$(ts) infer_head nc exit=$? -> $R/owl/probs_nc.csv ($(tail -1 logs/infer_nc.log))"
[ -f state/nc/failed.jsonl ] && echo "  nc failed cases: $(wc -l < state/nc/failed.jsonl) (state/nc/failed.jsonl)"
echo "$(ts) ALL DONE"
