#!/bin/bash
# two-way sync of the nc 1.5 mm rib emb15 between the A6000 (cg10k_nc/nc15/r15) and CT-rate (cg10k_best/nc_r15), no deletes.
B=~/Desktop/Tony/VLM/OWL/cg10k_nc; CT=daikso@192.168.0.189
while :; do
  rsync -a --exclude '*.tmp*' $CT:OWL/cg10k_best/nc_r15/emb15/ $B/nc15/r15/emb15/ 2>/dev/null
  rsync -a --exclude '*.tmp*' $B/nc15/r15/emb15/ $CT:OWL/cg10k_best/nc_r15/emb15/ 2>/dev/null
  grep -q NC15_DONE $B/nc15.log && ssh $CT "grep -q CT_RIBS_NC_DONE OWL/cg10k_best/nc_r15/ct_worker.log" && break
  sleep 120
done; echo SYNC_R15_DONE
