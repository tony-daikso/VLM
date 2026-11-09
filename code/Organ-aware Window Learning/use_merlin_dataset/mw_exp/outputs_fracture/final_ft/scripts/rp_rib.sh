#!/bin/bash
# fracture rib/pelvis push (agent fracture_rp, 11-07 23:05): final_ft variants on top of spine15_add with rib_v3 (lev28 pool, 28 seeds).
# CPU only (A6000). Selection across variants = stacker train CV only (summary.json selected_train_cv).
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/final_ft; PY=~/miniconda3/envs/radar/bin/python; F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture
SP15="--add FT_sp15=$F/spine15_ft@vert"; RV3=$F/rib_v3/cand/lev28
run(){ t=$1; shift; [ -f $t/summary.json ] && return; OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag $t "$@" > $t.log 2>&1; echo "$(date "+%m-%d %H:%M") $t rc=$?" >> rp_chain.log; }
run rp_rv3add $SP15 --add RV3=$RV3@rib
run rp_rv3repl $SP15 --rib $RV3
