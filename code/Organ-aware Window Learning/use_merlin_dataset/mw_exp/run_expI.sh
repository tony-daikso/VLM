#!/bin/bash
# Exp I queue runner (CPU only). usage: NQ=3 bash run_expI.sh jobs.txt   (jobs.txt: "<variant> <seed>" per line)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
D=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/expI; L=$D/logs; mkdir -p $L $D/members
mapfile -t JOBS < "$1"
NQ=${NQ:-3}; TH=${TH:-4}
q() { i=$1; while [ $i -lt ${#JOBS[@]} ]; do set -- ${JOBS[$i]}
  v=$1; s=$2; tag=${v//+/-}
  if [ ! -f $D/members/oof_bone_E_${tag}_s$s.npz ] && [ ! -f $L/${tag}_s$s.running ]; then
    touch $L/${tag}_s$s.running
    OMP_NUM_THREADS=$TH $PY bone_expI.py --variant "$v" --seed-off $s --threads $TH > $L/${tag}_s$s.log 2>&1
    rm -f $L/${tag}_s$s.running
  fi
  i=$((i+NQ)); done; }
for k in $(seq 0 $((NQ-1))); do q $k & done
wait
echo done > $D/logs/QUEUE_DONE_$(basename $1 .txt)
