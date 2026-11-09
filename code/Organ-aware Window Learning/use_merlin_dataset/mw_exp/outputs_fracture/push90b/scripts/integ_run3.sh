#!/bin/bash
# push90b integration runner (A6000 CPU, 3 threads, nice 10). usage: integ_run3.sh TAG [final_ft3 args...]
# default = --cvonly (train CV only, no val computed); FULL=1 integ_run3.sh TAG ... = final val pass (only for the selected tag).
# base inputs always added: FT_sp15 (spine15_ft@vert). skip if done; one try per tag; stop: touch push90b/STOP
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; cd $F/push90b; PY=~/miniconda3/envs/radar/bin/python
t=$1; shift; mkdir -p integ
if [ "$FULL" = 1 ]; then done_f=integ/$t/summary.json; md=""; lg=integ/$t.full.log; else done_f=integ/$t/summary_cv.json; md="--cvonly"; lg=integ/$t.log; fi
[ -f $done_f ] && exit 0; [ -f STOP ] && exit 0
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 10 $PY $F/push90b/scripts/final_ft3.py --tag $t $md --add FT_sp15=$F/spine15_ft@vert "$@" > $lg 2>&1
rc=$?; echo "$(date "+%m-%d %H:%M") integ${FULL:+ FULL} $t rc=$rc $(grep -o "\"selected_train_cv\": [0-9.]*" $done_f 2>/dev/null)" >> logs/push90b.log
