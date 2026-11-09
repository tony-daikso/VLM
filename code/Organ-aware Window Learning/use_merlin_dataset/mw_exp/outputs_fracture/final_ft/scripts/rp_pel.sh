#!/bin/bash
# fracture rib/pelvis push (agent fracture_rp): final_ft variants with pelvis15 (PV15 = pelvis15/final_m400 20-seed pool) +/- rib_v3 (lev28).
# CPU only (A6000). Selection across ALL rp_* variants + spine15_add = stacker train CV only.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/final_ft; PY=~/miniconda3/envs/radar/bin/python; F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture
SP15="--add FT_sp15=$F/spine15_ft@vert"; RV3=$F/rib_v3/cand/lev28; PV=$F/pelvis15/final_m400
run(){ t=$1; shift; [ -f $t/summary.json ] && return; OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag $t "$@" > $t.log 2>&1; echo "$(date "+%m-%d %H:%M") $t rc=$?" >> rp_chain.log; }
run rp_pv15add $SP15 --add PV15=$PV@pel
run rp_pv15repl $SP15 --pelvis $PV
run rp_both_aa $SP15 --add RV3=$RV3@rib --add PV15=$PV@pel
run rp_both_ar $SP15 --add RV3=$RV3@rib --pelvis $PV
run rp_both_ra $SP15 --rib $RV3 --add PV15=$PV@pel
run rp_both_rr $SP15 --rib $RV3 --pelvis $PV
echo RP_PEL_DONE >> rp_chain.log
