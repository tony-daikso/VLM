#!/bin/bash
# fracture final combo: expI + J_add + L(repl-rl03-lrc) [+ Exp O pelvis heads]; config chosen by stacker train 5-fold CV only.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY="nice -n 19 env OMP_NUM_THREADS=2 /home/advantech/miniconda3/envs/radar/bin/python"
F=../outputs_fracture/final_combo; M=$F/members
O=../outputs_fracture/expL/oof/full
J10="../outputs_vert15/final_eval/oof/full/oof_v15_add_s*.npz"; EI=../outputs_fracture/bones/expI/groups/ens_greedy_all.npz
BASE="--member expI=$EI --member J_add=$J10 --member L=$O/oof_L_L_repl-rl03-lrc_s*.npz"
REFS="--extra-ref J_add10=$J10 --extra-ref L_final851=$M/expL_final_stack.npz --no-feat --no-e3 --n-boot 2000"
run(){ n=$1; shift; $PY $F/scripts/final_stack_combo.py $BASE "$@" $REFS --title "骨折最終整合 $n" --out $F/$n > $F/logs/$n.log 2>&1; echo "$n done"; }
run C0_ILJ &
run C1_ILJ_pel --member U_org_pel=$M/U_org_pel.npz &
run C2_ILJ_pel_out --member U_org_pel=$M/U_org_pel.npz --member U_org_out=$M/U_org_out.npz &
run C3_ILJ_pel_all --member U_org_pel=$M/U_org_pel.npz --member U_pel=$M/U_pel.npz --member U_org_out=$M/U_org_out.npz &
wait; grep -H "stacker 以" $F/C*/report.md; echo COMBO_DONE
