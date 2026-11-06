#!/bin/bash
# CG10k 1.5 mm vertebra embeddings (experiment J encoder, TS Dataset292) for main + delay. Resumable.
#  1) waits until the cg10k OWL run is finished: no run_all.sh, no cg10k_run.py, probs_main.csv exists (3 checks in a row)
#  2) per set: ids = NIfTIs that have a 3 mm Dataset297 mask in owl/<set>/masks -> N_W GPU workers of
#     src_v15/vert15_full.py (each life max MAXC cases, recycled until DONE_w*/STOP) -> /media/daikso/CTDATA/cg10k/owl_v15/<set>/
# usage: tmux new -d -s cg10k_v15 "bash ~/OWL/cg10k_infer/run_v15.sh 2>&1 | tee -a ~/OWL/cg10k_infer/logs/v15_run.log"
# stop:   touch /media/daikso/CTDATA/cg10k/owl_v15/STOP   (workers finish their current case; loop exits)
# resume: rm STOP; per set rm emb15/*.tmp* seg15/*.tmp* DONE_w*; rm claims/<id> without emb15/<id>.npz; rerun this script
N_W=${N_W:-2}; MAXC=${MAXC:-200}; THREADS=${THREADS:-2}
D=/home/daikso/OWL/cg10k_infer; PY=/home/daikso/miniconda3/envs/radar/bin/python
R=/media/daikso/CTDATA/cg10k; O=$R/owl_v15
ts() { date "+%F %T"; }
cnt() { ls "$1" 2>/dev/null | grep -c '^[^.]*\.npz$'; }
mkdir -p $O $D/logs
n_ok=0; last=""
while [ $n_ok -lt 3 ]; do
  if pgrep -f "cg10k_infer/run_all.sh" >/dev/null || pgrep -f "src/cg10k_run.py" >/dev/null || [ ! -f $R/owl/probs_main.csv ]; then
    n_ok=0
    h=$(date +%H)
    if [ "$h" != "$last" ]; then last=$h
      echo "$(ts) waiting for cg10k run_all (main bones done: $(cnt $R/owl/main/emb_bones/bone_L400_W1800), delay: $(cnt $R/owl/delay/emb_bones/bone_L400_W1800))"
    fi
  else
    n_ok=$((n_ok+1))
  fi
  sleep 60
done
echo "$(ts) cg10k run_all finished; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
[ -f $O/STOP ] && { echo "$(ts) $O/STOP exists - exit"; exit 1; }
for s in main delay; do
  IN=$R/nifti_3mm; [ $s = delay ] && IN=$R/nifti_3mm_delay
  M=$R/owl/$s/masks; OUT=$O/$s
  [ -d $M ] || { echo "$(ts) $s: no mask dir, skip"; continue; }
  mkdir -p $OUT
  ls $IN | grep '\.nii\.gz$' | grep -v tmp | sed 's/\.nii\.gz$//' | sort | while read c; do [ -f $M/$c.nii.gz ] && echo $c; done > $OUT/ids.txt
  echo "$(ts) $s: $(ls $IN | grep -c '\.nii\.gz$') NIfTI, $(wc -l < $OUT/ids.txt) with 3 mm mask, already done $(cnt $OUT/emb15)"
  pids=()
  for i in $(seq 0 $((N_W-1))); do
    ( cd $D/src_v15
      while [ ! -f $O/STOP ] && [ ! -f $OUT/DONE_w$i ]; do
        OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          $PY -u vert15_full.py extract --ids $OUT/ids.txt --out $OUT --mask-dirs $M --ct-local $IN \
          --device cuda --wid $i --max-cases $MAXC --threads $THREADS < /dev/null
        sleep 2
      done ) >> $D/logs/v15_${s}_w$i.log 2>&1 &
    pids+=($!); echo "$(ts) $s worker w$i loop pid $!"; sleep 20
  done
  wait "${pids[@]}"
  echo "$(ts) $s finished: emb15 $(cnt $OUT/emb15) / $(wc -l < $OUT/ids.txt), seg15 $(cnt $OUT/seg15), failed.txt lines $(cat $OUT/failed.txt 2>/dev/null | wc -l)"
  [ -f $O/STOP ] && { echo "$(ts) stopped by STOP"; exit 0; }
done
echo "$(ts) ALL DONE"
