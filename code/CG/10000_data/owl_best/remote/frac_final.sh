#!/bin/bash
# finalize fracture: r18 s1 rerun on A6000 CPU (exact; the Spark-CPU run differed by up to 1.5 logit), L s2 on A6000;
# L s3/s4 from Spark GPU (L is GPU-trained everywhere, non-deterministic); then Exp I ensemble and C3 stack.
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python
$P cg_frac.py expI r18 1 &
until [ -f frac/L/oof/full/oof_L_L_repl-rl03-lrc_s2.npz ]; do sleep 30; done
sleep 20; for p in $(pgrep -f "cg_frac.py L \|_run_L_oof"); do kill $p 2>/dev/null; done
cp frac/L/oof/full_spark/*.npz frac/L/oof/full/
wait
$P cg_frac.py ens > frac/logs/ens.log 2>&1
$P cg_frac.py stack > frac/logs/stack.log 2>&1
echo FRAC_DONE
