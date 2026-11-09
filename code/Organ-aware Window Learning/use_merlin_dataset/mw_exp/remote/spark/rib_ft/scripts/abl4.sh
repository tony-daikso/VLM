#!/bin/bash
cd ~/OWL/rib_ft
for c in "h256:--hid 256" "h1024:--hid 1024" "d5:--hid 512 --drop 0.5" "remb:--hid 512 --ribemb" "e60:--hid 512 --epochs 60 --lr 5e-4" "wd1:--hid 512 --wd 0.1"; do
  n=${c%%:*}; args=${c#*:}
  ~/radar_env/bin/python scripts/rib_probe2.py labels.csv feats3_t0 runs/t0_f3_$n --seeds 3 --epochs 30 $args 2>&1 | grep -E "Error|Trace"
  ~/radar_env/bin/python scripts/cmp.py runs/t0_f3_$n
done
