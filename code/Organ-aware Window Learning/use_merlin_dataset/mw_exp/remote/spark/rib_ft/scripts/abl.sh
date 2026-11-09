#!/bin/bash
cd ~/OWL/rib_ft
for c in "s1:--blocks,s1" "s12:--blocks,s1,s2" "s234:--blocks,s2,s3,s4" "lse:--pool,lse" "remb:--ribemb" "h64:--hid,64,--wd,5e-2"; do
  n=${c%%:*}; args=$(echo ${c#*:} | sed "s/,--/ --/g; s/\([a-z0-9]\),\([a-z0-9]\)/\1,\2/g")
  args=$(echo ${c#*:} | tr "," " " | sed "s/--blocks \([a-z0-9]*\) \([a-z0-9]*\) \([a-z0-9]*\)/--blocks \1,\2,\3/; s/--blocks \(s[0-9]\) \(s[0-9]\)\b/--blocks \1,\2/")
  ~/radar_env/bin/python scripts/rib_probe.py labels.csv feats runs/abl_$n --seeds 1 --epochs 30 $args 2>&1 | grep -E "PROBE|Error"
done
