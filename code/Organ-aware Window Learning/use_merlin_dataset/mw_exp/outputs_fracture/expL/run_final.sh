#!/bin/bash
# Exp L final evaluation (full train, 20,355 rib emb15). train/val only; val AUROC not used for selection.
# stop: touch outputs_fracture/expL/final/STOP (checked between steps); kill: pkill -f "rib15_probe.py --stage oof"
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY="nice -n 19 env OMP_NUM_THREADS=3 /home/advantech/miniconda3/envs/radar/bin/python"
L=../outputs_fracture/expL; F=$L/final; mkdir -p $F/logs
LOCK=../outputs_final/A6000_GPU_LOCK
st(){ [ -f $F/STOP ] && { echo "STOPPED at $1 $(date)"; rm -f $LOCK; exit 0; }; }
echo "START $(date)"
st build
$PY rib15_probe.py --stage build --out $L > $F/logs/build.log 2>&1 || { echo "BUILD_FAIL $(date)"; exit 1; }
tail -1 $F/logs/build.log; echo "BUILD_DONE $(date)"
st oof
echo "expL_final (rib15_probe oof, small GPU) since $(date); stop: touch outputs_fracture/expL/final/STOP" > $LOCK
run(){ $PY rib15_probe.py --stage oof --out $L --variants "$1" --seeds 0 1 2 3 4 --device cuda --threads 3 > $F/logs/oof_$2.log 2>&1; }
run L_add+rl03 add_rl03 & run L_repl+rl03+lrc repl_rl03_lrc & run L_repl+rl03 repl_rl03 & run Jadd jadd &
wait
rm -f $LOCK
grep -h "^OOF " $F/logs/oof_*.log; echo "OOF_DONE $(date)"
st eval
O=$L/oof/full
$PY rib15_eval.py --ref "Jadd=$O/oof_L_Jadd_s*.npz" --ref "G3=$O/oof_L_G3_s0.npz" \
  --model "L_add-rl03=$O/oof_L_L_add-rl03_s*.npz" --model "L_repl-rl03-lrc=$O/oof_L_L_repl-rl03-lrc_s*.npz" \
  --model "L_repl-rl03=$O/oof_L_L_repl-rl03_s*.npz" --common --title "實驗 L 最終（全 train，5 seed）" --out $F/lrep > $F/logs/eval.log 2>&1
echo "LREP_DONE $(date)"
J10="../outputs_vert15/final_eval/oof/full/oof_v15_add_s*.npz"
EI=../outputs_fracture/bones/expI/groups/ens_greedy_all.npz
for v in L_add-rl03 L_repl-rl03-lrc L_repl-rl03; do
  st stack_$v
  $PY final_stack.py --member expI=$EI --member "J_add=$J10" --member "L=$O/oof_L_${v}_s*.npz" --no-feat --no-e3 \
    --extra-ref "J_add10=$J10" --extra-ref expI=$EI --n-boot 2000 \
    --title "實驗 L 最終：I 集成＋J add＋$v（vs J add .847）" --out $F/SL_$v > $F/logs/stack_$v.log 2>&1
done
echo "== stacker train CV (selection) =="; grep -H "stacker train" $F/SL_*/report.md
echo "ALL_DONE $(date)"
