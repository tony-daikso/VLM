#!/bin/bash
# pelvis15_ft tier lane (agent pelvis15_ft, 11-08). usage: tier_run.sh HOSTTAG DATASET RUN [extra pv15_ft args]
# tier = pelvis15 ids_tier (train 309 pelvis+ / 1,500 neg), 5 sitewise folds, held-out tier rows scored (t_pel), 1 seed, no val.
# resumable (per-epoch ckpt); at most 2 tries per run. stop: touch <base>/runs/STOP (after current epoch)
H=$1; DS=$2; n=$3; shift 3
if [ $H = a6 ]; then B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft; P=$B/../pelvis15; ROI=$P/roi_tier; SRC=~/Desktop/Tony/VLM/OWL/mw_exp/src
  TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset${DS}_*/nnUNet*)
else B=/media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft; P=/media/daikso/Tony/VLM/data/CT/merlin/pelvis15; ROI=$P/roi; SRC=~/OWL/mw_exp/src
  TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset${DS}_*/nnUNet* /media/daikso/Tony/VLM/models/ts/Dataset${DS}_*/nnUNet* 2>/dev/null | head -1); fi
cd $B; PY=~/miniconda3/envs/radar/bin/python; mkdir -p runs logs
for try in 1 2; do
  [ -f runs/$n/summary.json ] && break; [ -f runs/STOP ] && break
  echo "$(date '+%m-%d %H:%M') $n start try $try" >> logs/tier.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_ft.py --labels $P/labels.csv --roi $ROI \
    --out runs/$n --ts $TS --src $SRC --tune --pool-ids $P/ids_tier.txt --side-csv $P/pel_side.csv --fixed --maxshape 176,144,240 \
    --threads 3 "$@" >> logs/$n.log 2>&1
  echo "$(date '+%m-%d %H:%M') $n rc=$? $(grep 'PV15FT train OOF' logs/$n.log)" >> logs/tier.log
done
