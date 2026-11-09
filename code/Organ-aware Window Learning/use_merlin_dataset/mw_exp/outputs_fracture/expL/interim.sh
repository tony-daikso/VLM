#!/bin/bash
# Exp L interim (tier0) evaluation: build matrix -> OOF variants (train tier0 subset, 3 seeds) -> report
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY="nice -n 19 env OMP_NUM_THREADS=4 /home/advantech/miniconda3/envs/radar/bin/python"
L=../outputs_fracture/expL
$PY rib15_probe.py --stage build --out $L > $L/logs/build_interim.log 2>&1 || exit 1
V="Jadd G3 L_add L_repl L_add+rmax L_repl+rmax L_add+rmm L_repl+rl03 L_repl+rl1 L_repl+rl03+lrc L_add+rl03 G3+rl03"
$PY rib15_probe.py --stage oof --out $L --subset-file ../outputs_ribs15/ids_prio.csv --variants $V --seeds 0 1 2 --threads 4 > $L/logs/oof_interim.log 2>&1
grep "OOF " $L/logs/oof_interim.log
O=$L/oof/tier0
args="--ref Jadd=$O/oof_L_Jadd_s*.npz --ref G3=$O/oof_L_G3_s*.npz"
for v in L_add L_repl L_add-rmax L_repl-rmax L_add-rmm L_repl-rl03 L_repl-rl1 L_repl-rl03-lrc L_add-rl03 G3-rl03; do args="$args --model $v=$O/oof_L_${v}_s*.npz"; done
$PY rib15_eval.py $args --ids-file ../outputs_ribs15/ids_prio.csv --common --title "實驗 L 中期（tier0 子集，3 seed）" --out $L/interim > $L/logs/eval_interim.log 2>&1
echo INTERIM_DONE
