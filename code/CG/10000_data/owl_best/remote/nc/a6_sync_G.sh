#!/bin/bash
# pull the CT-rate nc Exp G emb_ribs (ct_expG_nc.sh) into cg10k_nc/data/nc/emb_ribs every 5 min (no deletes).
B=~/Desktop/Tony/VLM/OWL/cg10k_nc; CT=daikso@192.168.0.189
while :; do
  for w in baseline soft_tissue_L40_W400 bone_L400_W1800; do mkdir -p $B/data/nc/emb_ribs/$w
    rsync -a --exclude '*.tmp*' $CT:OWL/cg10k_best/ncG/emb_ribs/$w/ $B/data/nc/emb_ribs/$w/ 2>/dev/null; done
  ssh $CT "grep -q CT_EXPG_NC_DONE OWL/cg10k_best/ncG/expG.log" && [ -n "$last" ] && break
  ssh $CT "grep -q CT_EXPG_NC_DONE OWL/cg10k_best/ncG/expG.log" && last=1
  sleep 300
done; echo SYNC_G_DONE
