#!/bin/bash
cd ~/OWL/rib_ft
for c in "s0:--blocks s0" "s1:--blocks s1" "s01h128:--blocks s0,s1 --hid 128" "s01d5:--blocks s0,s1 --drop 0.5" "s01wd:--blocks s0,s1 --wd 0.1" "s01h512:--blocks s0,s1 --hid 512" "s01lr3:--blocks s0,s1 --lr 3e-4 --epochs 60"; do
  n=${c%%:*}; args=${c#*:}
  ~/radar_env/bin/python scripts/rib_probe2.py labels.csv feats2 runs/p3_$n --seeds 1 --epochs 30 $args 2>&1 | grep -E "Error|Trace"
  ~/radar_env/bin/python scripts/cmp.py runs/p3_$n
done
