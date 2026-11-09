#!/bin/bash
# 4090 self-running driver for the focal-steatosis CNN (agent focal_steatosis_cnn, 2026-11-06).
# waits for work/PUSH_DONE (Mac relay) -> pack memmaps -> configs x seeds (5 folds + full each).
# caps: GPU <= 7.5 GB (per-process fraction), torch 2 threads, OMP/MKL 2, nice 10.  never touches ~/OWL/gs15.
# stop: touch ~/OWL/fsc/STOP (checked between models: finishes the current model, ~3-6 min); resume: rerun this script
# (finished pred_*.csv are skipped; pack is skipped if work/meta.csv exists).
cd ~/OWL/fsc || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/L/venv/bin/python
while [ ! -f work/PUSH_DONE ]; do [ -f STOP ] && exit 0; sleep 60; done
[ -f work/meta.csv ] || nice -n 10 $PY scripts/fsc_pack.py work crops > logs/pack.log 2>&1 || exit 1
declare -A ARGS=( [A]="--arch 3d --width 16 --epochs 20 --spe 4000" [B]="--arch 3d --width 24 --epochs 30 --spe 6000" [C]="--arch aniso --width 16 --epochs 20 --spe 4000" )
for s in 0 1 2; do for c in A B C; do
  [ -f STOP ] && { echo "STOPPED $(date +%T)"; exit 0; }
  nice -n 10 $PY scripts/fsc_train.py --work work --out runs/$c --seed $s ${ARGS[$c]} >> logs/train_${c}_s$s.log 2>&1
  echo "$c s$s rc=$? $(date +%T)"
done; done
echo ALL_TRAIN_DONE $(date +%T); touch ALL_TRAIN_DONE
