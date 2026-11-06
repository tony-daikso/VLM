#!/bin/bash
# CG10k OWL (E3_late expB) full run on CT-rate. Resumable: re-running skips finished cases (emb files exist).
#   phase 1: N_CPU CPU workers (nice 10, idle I/O class, CUDA hidden) while experiment J holds the 3080
#   phase 2: once every vert15_full.py process is gone (J finished/stopped) -> CPU workers told to stop after their
#            current case, N_GPU GPU workers on the 3080
#   sets (priority order): main = nifti_3mm, delay = nifti_3mm_delay. A set is complete when IN_DIR/.SYNC_DONE
#   exists (written from the Mac by mark_sync_done.sh) or no new file arrived for IDLE_H hours.
#   phase 3: infer_head.py -> /media/daikso/CTDATA/cg10k/owl/probs_main.csv, probs_delay.csv (+ per-seed columns)
# usage: tmux new -d -s cg10k 'bash ~/OWL/cg10k_infer/run_all.sh 2>&1 | tee -a ~/OWL/cg10k_infer/logs/run_all.log'
# stop everything: touch ~/OWL/cg10k_infer/state/STOP   (workers exit after their current case)
N_CPU=${N_CPU:-0}; CPU_THREADS=${CPU_THREADS:-4}
N_GPU=${N_GPU:-3}; GPU_THREADS=${GPU_THREADS:-4}; RECYCLE=${RECYCLE:-300}; J_WAIT=${J_WAIT:-1}
IDLE_H=${IDLE_H:-6}
D=/home/daikso/OWL/cg10k_infer; PY=/home/daikso/miniconda3/envs/radar/bin/python
R=/media/daikso/CTDATA/cg10k
SETS="--set main=$R/nifti_3mm=$R/owl/main --set delay=$R/nifti_3mm_delay=$R/owl/delay"
cd $D; mkdir -p logs state "$R/nifti_3mm_delay"
ts() { date '+%F %T'; }
rm -f state/STOP_cpu* state/STOP_gpu*
[ -f state/STOP ] && { echo "$(ts) state/STOP exists - remove it to run"; exit 1; }

pids=()
for i in $(seq 0 $((N_CPU-1))); do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS \
    nice -n 10 ionice -c3 $PY -u src/cg10k_run.py $SETS --device cpu --threads $CPU_THREADS --tag cpu$i \
    --watch --idle-exit-h $IDLE_H >> logs/cpu$i.log 2>&1 < /dev/null &
  pids+=($!); echo "$(ts) started cpu$i pid $!"
done

# wait until experiment J is gone for 3 consecutive checks (its bash loops also match vert15_full.py)
n_free=0; [ "$J_WAIT" = 0 ] && n_free=3
while [ $n_free -lt 3 ]; do
  if pgrep -f vert15_full.py > /dev/null; then n_free=0; else n_free=$((n_free+1)); fi
  # all CPU workers already finished everything -> no need for the GPU
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  if [ $N_CPU -gt 0 ] && [ $alive -eq 0 ]; then echo "$(ts) CPU workers finished before J ended"; break; fi
  sleep 60
done

alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
if [ $N_CPU -eq 0 ] || [ $alive -eq 1 ]; then
  echo "$(ts) experiment J gone; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
  for i in $(seq 0 $((N_CPU-1))); do touch state/STOP_cpu$i; done
  for i in $(seq 0 $((N_GPU-1))); do
    # 11-06: GPU steps serialized by flock (state/gpu.lock) -> no OOM between workers; each worker process is
    #        recycled every RECYCLE cases (exit 10 -> restart; long-lived processes got ~2x slower per case)
    ( while :; do
        OMP_NUM_THREADS=$GPU_THREADS MKL_NUM_THREADS=$GPU_THREADS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          $PY -u src/cg10k_run.py $SETS --device cuda --threads $GPU_THREADS --tag gpu$i \
          --watch --idle-exit-h $IDLE_H --gpu-lock $D/state/gpu.lock --max-cases $RECYCLE >> logs/gpu$i.log 2>&1 < /dev/null
        rc=$?; [ $rc -eq 10 ] || { echo "$(ts) gpu$i worker exit rc=$rc"; break; }
        [ -f state/STOP ] || [ -f state/STOP_gpu$i ] && break
      done ) &
    pids+=($!); echo "$(ts) started gpu$i loop pid $!"
    sleep 10   # stagger model loading
  done
fi
wait
echo "$(ts) all workers exited; done main=$(ls $R/owl/main/emb_bones/bone_L400_W1800 2>/dev/null | grep -c 'npz$') delay=$(ls $R/owl/delay/emb_bones/bone_L400_W1800 2>/dev/null | grep -c 'npz$')"
[ -f state/STOP ] && { echo "$(ts) stopped by state/STOP - skip inference"; exit 0; }
for s in main delay; do
  [ -d $R/owl/$s/emb/baseline ] || continue
  $PY -u src/infer_head.py --model $D/model/E3_late.pt --emb-dir $R/owl/$s/emb --bone-emb-dir $R/owl/$s/emb_bones \
    --per-seed --out $R/owl/probs_$s.csv > logs/infer_$s.log 2>&1
  echo "$(ts) infer_head $s exit=$? -> $R/owl/probs_$s.csv ($(tail -1 logs/infer_$s.log))"
  [ -f state/$s/failed.jsonl ] && echo "  $s failed cases: $(wc -l < state/$s/failed.jsonl) (state/$s/failed.jsonl)"
done
echo "$(ts) ALL DONE"
