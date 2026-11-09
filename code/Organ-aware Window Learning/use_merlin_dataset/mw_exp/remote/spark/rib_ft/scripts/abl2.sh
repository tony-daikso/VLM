#!/bin/bash
cd ~/OWL/rib_ft
for c in "all:" "s01:--blocks s0,s1" "remb:--ribemb" "lse:--pool lse" "s012:--blocks s0,s1,s2"; do
  n=${c%%:*}; args=${c#*:}
  ~/radar_env/bin/python scripts/rib_probe2.py labels.csv feats2 runs/p2_$n --seeds 1 --epochs 30 $args 2>&1 | grep -E "PROBE|Error|Trace"
  ~/radar_env/bin/python scripts/cmp.py runs/p2_$n
done
