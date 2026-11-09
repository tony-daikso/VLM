#!/bin/bash
cd ~/OWL/rib_ft
for c in "remb:--hid 512 --ribemb --epochs 60 --lr 5e-4" "h1ke60:--hid 1024 --epochs 60 --lr 5e-4" "e60wd:--hid 512 --epochs 60 --lr 5e-4 --wd 0.1"; do
  n=${c%%:*}; args=${c#*:}
  ~/radar_env/bin/python scripts/rib_probe2.py labels.csv feats3_t0 runs/t0_f3_$n --seeds 3 --epochs 30 $args 2>&1 | grep -E "Error"
  ~/radar_env/bin/python scripts/cmp.py runs/t0_f3_$n
done
