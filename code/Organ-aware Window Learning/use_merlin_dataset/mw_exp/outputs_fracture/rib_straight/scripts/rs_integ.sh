#!/bin/bash
# rib_straight integration (agent rib_straight; A6000 CPU): final_ft variants on top of rp_both_aa (SP15 + RV3@rib + PV15@pel).
# RS = straightened-tile probe pool (combine3 of prod runs, col ft_side); MIX = one rib head from RV3 lev28 runs + RS runs.
# optional PF dir (pelvis15_ft final) as 1st arg -> extra variants. Selection = stacker train CV only (rp_select.py).
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; cd $F/final_ft; PY=~/miniconda3/envs/radar/bin/python; PF=$1
SP15="--add FT_sp15=$F/spine15_ft@vert"; RV3=$F/rib_v3/cand/lev28; PV=$F/pelvis15/final_m400; RS=$F/rib_straight/cand/rs; MIX=$F/rib_straight/cand/mix
run(){ t=$1; shift; [ -f $t/summary.json ] && return; OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY scripts/final_ft.py --tag $t "$@" > $t.log 2>&1; echo "$(date "+%m-%d %H:%M") $t rc=$?" >> $F/rib_straight/logs/integ.log; }
run rs_add  $SP15 --add RV3=$RV3@rib --add RS=$RS@rib --add PV15=$PV@pel
run rs_repl $SP15 --add RS=$RS@rib --add PV15=$PV@pel
run rs_mix  $SP15 --add RVRS=$MIX@rib --add PV15=$PV@pel
if [ -n "$PF" ]; then
  run rs_add_pf  $SP15 --add RV3=$RV3@rib --add RS=$RS@rib --add PV15=$PV@pel --add PF=$PF@pel
  run rp_both_aa_pf $SP15 --add RV3=$RV3@rib --add PV15=$PV@pel --add PF=$PF@pel
fi
echo RS_INTEG_DONE >> $F/rib_straight/logs/integ.log
