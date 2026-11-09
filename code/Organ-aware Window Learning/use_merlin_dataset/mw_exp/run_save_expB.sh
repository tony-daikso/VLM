#!/bin/bash
# 2026-11-05: refit E3_late expB (standardize, 17 bone units, fracture MIL max) on CPU with the HP chosen in
# results/expB/E3_late_grid.json (lr=1e-3, wd=1e-4), 3 seeds, test ids dropped; save heads for inference.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
M=../outputs_final/models/E3_late_expB
nice -n 10 $PY probe.py --config ../config_final.yaml --results-subdir expB_savemodel --standardize --drop-test \
  --only E3_late --hp 0.001 0.0001 --seeds 3 --save-model $M || exit 1
R=../outputs_final/results
nice -n 10 $PY verify_saved_head.py --model $M/E3_late.pt \
  --matrix "$R/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz" \
  --train-npz $R/expB_savemodel/E3_late.npz --ref-npz $R/expB/E3_late.npz --out $M/val_reproduce.csv || exit 1
echo SAVE_EXPB_DONE
