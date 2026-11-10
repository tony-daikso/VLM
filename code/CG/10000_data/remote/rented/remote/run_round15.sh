#!/bin/bash
# ON the rented L40S: one v15 round. usage: run_round15.sh K SET N_W THREADS  (ids in state/v15_round_K.txt)
# inputs in/v15/SET/{ct,masks}, outputs out/v15/SET/{emb15,seg15} (unchanged src_v15/vert15_full.py --ct-local)
# writes state/v15_round_K.done when all workers exit. STOP: kill the workers (coordinator does it).
K=$1; S=$2; N_W=${3:-4}; TH=${4:-3}
G=/root/g4090; PY=/root/radar_env/bin/python; O=$G/out/v15/$S; IDS=$G/state/v15_round_$K.txt
mkdir -p $O/claims $G/logs
rm -f $G/state/v15_round_$K.done $O/emb15/*.tmp* $O/seg15/*.tmp* $O/DONE_w*
for c in $(cat $IDS); do [ -f $O/emb15/$c.npz ] || rm -f $O/claims/$c; done
cd $G/code/src_v15
for i in $(seq 0 $((N_W-1))); do
  OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u vert15_full.py extract --ids $IDS --out $O --mask-dirs $G/in/v15/$S/masks --ct-local $G/in/v15/$S/ct \
    --device cuda --wid $i --threads $TH >> $G/logs/v15_w$i.log 2>&1 < /dev/null &
  sleep 2
done
wait
date +%s > $G/state/v15_round_$K.done
