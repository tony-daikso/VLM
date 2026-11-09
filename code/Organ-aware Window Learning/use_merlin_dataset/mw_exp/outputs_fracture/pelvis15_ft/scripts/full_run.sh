#!/bin/bash
# pelvis15_ft FULL run lane (agent pelvis15_ft, 11-08). usage: full_run.sh HOST CFG "EXTRA ARGS" fold:seed [fold:seed ...]
# pool = tierA & t_pel known (pool_ids.txt: 309 pos + 4,000 neg), Dataset297, final mode, --no-val (val + non-pool train rows are
# filled later on the CT-rate 3080 with --score). out: runs/full_<CFG>_s<seed>/fold<k>.npz + model_f<k>_s<seed>.pt
# resumable (per-epoch ckpt, finished folds skipped); at most 2 tries per job; a job failing twice stops the lane.
# stop: touch <base>/runs/STOP (after the current epoch)
H=$1; CFG=$2; EX=$3; shift 3
case $H in
 a6) B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft; P=$B/../pelvis15; ROI=$B/roi_pool; SRC=~/Desktop/Tony/VLM/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; AMP="--workers 4";;
 ct) B=/media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft; P=$B/../pelvis15; ROI=$P/roi; SRC=~/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; AMP="--amp bf16 --vram-gb 9 --workers 4";;
 sp) B=~/Desktop/Tony/OWL/pelvis15_ft; P=$B/in; ROI=$B/roi; SRC=~/Desktop/Tony/OWL/mw_exp/src; PY=~/radar_env/bin/python; AMP="--amp bf16 --vram-gb 30 --workers 4";;
esac
TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset297_*/nnUNet* | head -1)
cd $B; mkdir -p runs logs
for job in "$@"; do
  k=${job%:*}; s=${job#*:}; o=runs/full_${CFG}_s$s; ok=0
  for try in 1 2; do
    [ -f $o/fold$k.npz ] && { ok=1; break; }; [ -f runs/STOP ] && exit 0
    echo "$(date '+%m-%d %H:%M') $H $o fold $k start try $try" >> logs/full.log
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_ft.py --labels $P/labels.csv --roi $ROI \
      --out $o --ts $TS --src $SRC --pool-ids $B/pool_ids.txt --side-csv $P/pel_side.csv --fixed --maxshape 176,144,240 \
      --threads 3 --no-val --folds $k --seed0 $s --seeds 1 $EX $AMP >> logs/full_${CFG}_s${s}_f$k.log 2>&1
    rc=$?; echo "$(date '+%m-%d %H:%M') $H $o fold $k rc=$rc $(grep "fold $k OOF" logs/full_${CFG}_s${s}_f$k.log | tail -1)" >> logs/full.log
    [ -f $o/fold$k.npz ] && { ok=1; break; }
  done
  [ $ok = 1 ] || { echo "$(date '+%m-%d %H:%M') $H job $job FAILED twice -> lane stops" >> logs/full.log; exit 1; }
done
echo "$(date '+%m-%d %H:%M') $H lane done: $*" >> logs/full.log
