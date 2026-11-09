#!/bin/bash
# push90 integration runner (A6000 CPU). usage: integ_run.sh TAG [final_ft2 args...]; base = rsP_mix_repl inputs, extra args appended.
# skip if done; one try per tag; logs push90/integ/<tag>.log; stop: touch push90/STOP
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; cd $F/push90; PY=~/miniconda3/envs/radar/bin/python
t=$1; shift
[ -f integ/$t/summary.json ] && exit 0; [ -f STOP ] && exit 0
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 nice -n 10 $PY scripts/final_ft2.py --tag $t --add FT_sp15=$F/spine15_ft@vert "$@" > integ/$t.log 2>&1
echo "$(date "+%m-%d %H:%M") integ $t rc=$? cv=$(grep -o "\"selected_train_cv\": [0-9.]*" integ/$t/summary.json 2>/dev/null)" >> logs/push90.log
