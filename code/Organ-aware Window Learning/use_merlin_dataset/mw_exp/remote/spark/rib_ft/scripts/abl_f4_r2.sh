#!/bin/bash
# fracture_round2 agent: feats4 (stage 0-2) vs feats3 (stage 0-1) on the same 1749 tier0 rows, chosen probe config. train OOF only.
cd ~/OWL/rib_ft
for n in f3m f4; do d=feats3_t0m; [ $n = f4 ] && d=feats4
  ~/radar_env/bin/python scripts/rib_probe2.py labels.csv $d runs/r2_t0_$n --seeds 3 --hid 512 --epochs 60 --lr 5e-4 > logs/r2_t0_$n.log 2>&1
  ~/radar_env/bin/python scripts/cmp.py runs/r2_t0_$n >> logs/r2_abl_f4.log 2>&1
done
echo ABL_DONE >> logs/r2_abl_f4.log
