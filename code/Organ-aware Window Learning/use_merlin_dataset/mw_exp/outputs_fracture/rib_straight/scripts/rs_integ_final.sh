#!/bin/bash
# rib_straight final integration (agent rib_straight; A6000 CPU): rib heads (RS15 = 15-seed tile probe; MIX15 = z-avg RV3 lev28 + RS15;
# MIX3 = z-avg RV3 + RS15 + RS6 [9 mm]) x pelvis heads (PV15 / P15FT = pelvis15_ft final from the pelvis agent).
# selection = stacker train CV only.
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; cd $F/final_ft; PY=~/miniconda3/envs/radar/bin/python
SP15="--add FT_sp15=$F/spine15_ft@vert"; RV3=$F/rib_v3/cand/lev28; PV=$F/pelvis15/final_m400; PF=$F/pelvis15_ft/final
RS=$F/rib_straight/cand/rs15; MIX=$F/rib_straight/cand/mix15; MIX3=$F/rib_straight/cand/mix3
run(){ t=$1; shift; [ -f $t/summary.json ] && return; OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag $t "$@" > $t.log 2>&1; echo "$(date "+%m-%d %H:%M") $t rc=$?" >> $F/rib_straight/logs/integ.log; }
run rs3_mix      $SP15 --add RVRS=$MIX3@rib --add PV15=$PV@pel
run rsP_mix_repl $SP15 --add RVRS=$MIX@rib --add P15FT=$PF@pel
run rsP_mix_add  $SP15 --add RVRS=$MIX@rib --add PV15=$PV@pel --add P15FT=$PF@pel
run rsP_add_repl $SP15 --add RV3=$RV3@rib --add RS=$RS@rib --add P15FT=$PF@pel
run rsP3_repl    $SP15 --add RVRS=$MIX3@rib --add P15FT=$PF@pel
echo RSF_INTEG_DONE >> $F/rib_straight/logs/integ.log
