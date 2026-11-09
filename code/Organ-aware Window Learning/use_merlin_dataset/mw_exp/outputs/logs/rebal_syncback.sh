#!/bin/bash
# A6000: every 10 min pull finished bone embeddings of A6000 cases from CT-rate outputs_from_a6000/emb_bones/
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs
R=daikso@192.168.0.189
W="baseline soft_tissue_L40_W400 bone_L400_W1800"
sync() { for w in $W; do rsync -a --ignore-existing --exclude '*.tmp.npz' $R:OWL/mw_exp/outputs_from_a6000/emb_bones/$w/ emb_bones/$w/; done; }
missing() { n=0; while read i; do for w in $W; do [ -f emb_bones/$w/$i.npz ] || { n=$((n+1)); break; }; done; done < a6000_mask_ids.txt; echo $n; }
while true; do
  sync; m=$(missing)
  la=$(pgrep -f "[e]xtract_bones.py" | wc -l); lc=$(ssh -o BatchMode=yes $R 'pgrep -f [e]xtract_bones.py | wc -l')
  echo "$(date '+%F %T') synced; A6000 ids missing=$m; procs A6000=$la CT-rate=$lc; per-window: $(for w in $W; do ls emb_bones/$w | grep -c 'npz$'; done | tr '\n' ' ')"
  if [ "$m" -eq 0 ] || { [ "$la" -eq 0 ] && [ "$lc" -eq 0 ]; }; then sleep 30; sync; echo "$(date '+%F %T') FINAL sync done; missing=$(missing)"; break; fi
  sleep 600
done
