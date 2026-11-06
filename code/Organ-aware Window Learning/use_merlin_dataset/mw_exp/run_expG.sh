#!/bin/bash
# Exp G (rib-level units): 10-seed OOF members on A6000 GPU, then seed groups + greedy ensemble (train OOF) + final stacks.
# Runs after emb_ribs is complete and merged into outputs/emb_ribs and `rib_probe.py --stage build` is done.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones
D=$B/final_G; O=$D/members; L=$D/logs; mkdir -p $O $L
JOBS=()
for cfg in "rib18 ref site_max" "rib18 excl_amb site_max" "rib18big ref site_max" "rib18 ref site_lse"; do
  for s in 0 1 2 3 4 5 6 7 8 9; do JOBS+=("$cfg $s"); done
done
NQ=${NQ:-3}
q() { i=$1; while [ $i -lt ${#JOBS[@]} ]; do set -- ${JOBS[$i]}
  f=$L/$2__$3-$1_s$4.log
  OMP_NUM_THREADS=2 $PY rib_probe.py --stage oof --ribset $1 --variant $2 --model $3 --seed-off $4 --device cuda --outdir $O > $f 2>&1
  i=$((i+NQ)); done; }
for k in $(seq 0 $((NQ-1))); do q $k & done
wait
echo done > $D/G_MEMBERS_DONE
# groups: G members + the Exp F members (symlinked) -> greedy ensemble on TRAIN OOF
mkdir -p $D/ens_all/members $D/ens_G/members
ln -sf $B/final_F1/members/*.npz $D/ens_all/members/
ln -sf $O/*.npz $D/ens_all/members/
ln -sf $O/*.npz $D/ens_G/members/
$PY bone_ens.py --dir $D/ens_all > $D/ens_all.log 2>&1
$PY bone_ens.py --dir $D/ens_G > $D/ens_G.log 2>&1
F1=$B/final_F1/groups/bone_ens.npz
for c in "ens_all:$D/ens_all/groups/bone_ens.npz" "rib18ref10:$D/ens_G/groups/ref__site_max-rib18__mean__all3.npz" \
         "rib18excl10:$D/ens_G/groups/excl_amb__site_max-rib18__mean__all3.npz"; do
  n=${c%%:*}; f=${c#*:}
  OMP_NUM_THREADS=3 $PY final_stack.py --member bone_$n=$f --extra-ref bone_ens_F1=$F1 \
    --title "實驗 G：單根肋骨單元（候選 $n）" --out $D/cand_$n > $D/cand_$n.log 2>&1 &
done
wait
echo done > $D/G_DONE
