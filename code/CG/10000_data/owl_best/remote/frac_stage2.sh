#!/bin/bash
# fracture stage 2: after all CG Exp G (3 mm single-rib) embeddings exist -> rib rows -> rib_matrix -> Exp I r18 (2 seeds)
# + L (rib15_probe L_repl+rl03+lrc, 5 seeds) -> then ens + stack (needs J_add 10 seeds and the Exp O heads).
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python
cnt(){ ls data/$1/emb_ribs/bone_L400_W1800 2>/dev/null | grep -v tmp | wc -l; }
until [ $(cnt main) -ge 6970 ] && [ $(cnt delay) -ge 810 ]; do sleep 120; done
$P cg_frac_rows.py data/main/emb_ribs "" frac/cg_main_ribs3.npz > frac/logs/ribrows.log 2>&1
$P cg_frac_rows.py data/delay/emb_ribs D_ frac/cg_delay_ribs3.npz >> frac/logs/ribrows.log 2>&1
$P cg_frac.py prep ribs >> frac/logs/ribrows.log 2>&1
( for s in 0 1; do $P cg_frac.py expI r18 $s; done; echo R18_DONE ) > frac/logs/chain_r18.log 2>&1 &
( for s in 0 1 2 3 4; do $P cg_frac.py L $s; done; echo L_DONE ) > frac/logs/chain_L.log 2>&1 &
wait
until grep -q V15_DONE frac/logs/chain_v15b.log; do sleep 60; done
$P cg_frac.py ens > frac/logs/ens.log 2>&1
$P cg_frac.py stack > frac/logs/stack.log 2>&1
echo FRAC_DONE
