#!/bin/bash
# Exp F part 1: seed-average groups, greedy ensemble (train OOF), candidate stacks (stacker by train CV)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
D=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/final_F1
G=$D/groups
$PY bone_ens.py --dir $D > $D/bone_ens.log 2>&1
for c in ref10:$G/ref__site_max__mean__all3.npz excl10:$G/excl_amb__site_max__mean__all3.npz ens:$G/bone_ens.npz; do
  n=${c%%:*}; f=${c#*:}
  OMP_NUM_THREADS=3 $PY final_stack.py --member bone_$n=$f --title "實驗 F：最終整合（候選 $n）" --out $D/cand_$n > $D/cand_$n.log 2>&1 &
done
OMP_NUM_THREADS=3 $PY final_stack.py --member ref10=$G/ref__site_max__mean__all3.npz --member excl10=$G/excl_amb__site_max__mean__all3.npz \
  --title "實驗 F：最終整合（候選 ref10+excl10）" --out $D/cand_ref10_excl10 > $D/cand_ref10_excl10.log 2>&1 &
wait
echo done > $D/STACKS_DONE
