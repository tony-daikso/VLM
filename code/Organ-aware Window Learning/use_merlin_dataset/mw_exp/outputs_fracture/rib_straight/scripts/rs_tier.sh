#!/bin/bash
# rib_straight tier lane (agent rib_straight, 11-08). usage: rs_tier.sh HOST DATASET RUN [extra rs_ft args]
# tier = in/ids_tier.txt (train rib+ 505 + 1,500 random rib-negatives), 5 sitewise folds, held-out tier rows scored (t_rib), 1 seed, no val.
# resumable (per-epoch ckpt + per-fold npz); at most 2 tries. stop: touch <base>/runs/STOP (after current epoch)
H=$1; DS=$2; n=$3; shift 3
case $H in
 a6) B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_straight; SRC=~/Desktop/Tony/VLM/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; EX="--workers 4";;
 ct) B=/media/daikso/Tony/VLM/data/CT/merlin/rib_straight; SRC=~/OWL/mw_exp/src; PY=~/miniconda3/envs/radar/bin/python; EX="--amp bf16 --vram-gb 5 --workers 3";;
 sp) B=~/Desktop/Tony/OWL/rib_straight; SRC=~/Desktop/Tony/OWL/mw_exp/src; PY=~/radar_env/bin/python; EX="--amp bf16 --vram-gb 24 --workers 4";;
esac
TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset${DS}_*/nnUNet* /media/daikso/Tony/VLM/models/ts/Dataset${DS}_*/nnUNet* 2>/dev/null | head -1)
cd $B; mkdir -p runs logs
for try in 1 2; do
  [ -f runs/$n/summary.json ] && break; [ -f runs/STOP ] && break
  echo "$(date '+%m-%d %H:%M') $H $n start try $try" >> logs/tier.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/rs_ft.py --labels in/labels.csv --tiles tiles \
    --out runs/$n --ts $TS --src $SRC --tune --pool-ids in/ids_tier.txt --side-csv in/rib_side.csv --threads 3 $EX "$@" >> logs/$n.log 2>&1
  echo "$(date '+%m-%d %H:%M') $H $n rc=$? $(grep 'RSFT train OOF' logs/$n.log | tail -1)" >> logs/tier.log
done
