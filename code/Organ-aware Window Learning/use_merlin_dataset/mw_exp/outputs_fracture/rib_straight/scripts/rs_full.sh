#!/bin/bash
# rib_straight FULL run lane (agent rib_straight, 11-08). usage: rs_full.sh HOST CFG DATASET "EXTRA ARGS" fold:seed [fold:seed ...]
# pool = all train rows with t_rib known; each epoch = all pool positives + --neg-epoch fresh negatives; final mode: held-out fold
# rows (all train) + val scored by each fold model; models saved. out: runs/full_<CFG>_s<seed>/fold<k>.npz + model_f<k>_s<seed>.pt
# resumable (per-epoch ckpt, finished folds skipped); at most 2 tries per job; a job failing twice stops the lane.
# stop: touch <base>/runs/STOP (after the current epoch)
H=$1; CFG=$2; DS=$3; EX=$4; shift 4
case $H in
 a6) B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_straight; SRC=~/Desktop/Tony/VLM/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; AMP="--workers 4";;
 ct) B=/media/daikso/Tony/VLM/data/CT/merlin/rib_straight; SRC=~/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; AMP="--amp bf16 --vram-gb 6 --workers 3";;
 sp) B=~/Desktop/Tony/OWL/rib_straight; SRC=~/Desktop/Tony/OWL/mw_exp/src; PY=~/radar_env/bin/python; AMP="--amp bf16 --vram-gb 24 --workers 4";;
esac
TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset${DS}_*/nnUNet* /media/daikso/Tony/VLM/models/ts/Dataset${DS}_*/nnUNet* 2>/dev/null | head -1)
cd $B; mkdir -p runs logs
for job in "$@"; do
  k=${job%:*}; s=${job#*:}; o=runs/full_${CFG}_s$s; ok=0
  for try in 1 2; do
    [ -f $o/fold$k.npz ] && { ok=1; break; }; [ -f runs/STOP ] && exit 0
    echo "$(date '+%m-%d %H:%M') $H $o fold $k start try $try" >> logs/full.log
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rs_ft.py --labels in/labels.csv --tiles tiles \
      --out $o --ts $TS --src $SRC --side-csv in/rib_side.csv --threads 3 --folds $k --seed0 $s --seeds 1 $EX $AMP >> logs/full_${CFG}_s${s}_f$k.log 2>&1
    rc=$?; echo "$(date '+%m-%d %H:%M') $H $o fold $k rc=$rc $(grep "fold $k OOF" logs/full_${CFG}_s${s}_f$k.log | tail -1)" >> logs/full.log
    [ -f $o/fold$k.npz ] && { ok=1; break; }
  done
  [ $ok = 1 ] || { echo "$(date '+%m-%d %H:%M') $H job $job FAILED twice -> lane stops" >> logs/full.log; exit 1; }
done
echo "$(date '+%m-%d %H:%M') $H lane done: $*" >> logs/full.log
