#!/bin/bash
# Exp F part 1: 10-seed OOF members on A6000 GPU (folds fixed; seed-off changes model init only)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
O=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/final_F1/members
L=$O/../logs; mkdir -p $L
JOBS=()
for s in 1 2 3 4 5 6 7 8 9; do JOBS+=("ref site_max mean all3 $s"); done
for s in 0 1 2 3 4 5 6 7 8 9; do JOBS+=("excl_amb site_max mean all3 $s"); done
for cfg in "site_max max all3" "site_max mean baseline" "milv_max mean all3" "site_max mean bone" "site_lse mean all3"; do
  for s in 0 1 2 3 4 5 6 7 8 9; do JOBS+=("ref $cfg $s"); done
done
NQ=5
q() { i=$1; while [ $i -lt ${#JOBS[@]} ]; do set -- ${JOBS[$i]}
  OMP_NUM_THREADS=2 $PY bone_label_exp.py --variant $1 --model $2 --pool $3 --win $4 --seed-off $5 --device cuda --threads 2 --outdir $O > $L/$1__$2__$3__$4_s$5.log 2>&1
  i=$((i+NQ)); done; }
for k in $(seq 0 $((NQ-1))); do q $k & done
wait
echo done > $O/../F1_DONE
