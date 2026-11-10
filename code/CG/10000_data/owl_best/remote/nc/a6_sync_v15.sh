#!/bin/bash
# two-way sync of the nc v15 emb15 + seg15 between the A6000 (cg10k_nc/nc15/v15) and CT-rate (cg10k_best/nc_v15), no deletes.
B=~/Desktop/Tony/VLM/OWL/cg10k_nc; CT=daikso@192.168.0.189
while :; do
  for d in emb15 seg15; do
    rsync -a --exclude '*.tmp*' $CT:OWL/cg10k_best/nc_v15/$d/ $B/nc15/v15/$d/ 2>/dev/null
    rsync -a --exclude '*.tmp*' $B/nc15/v15/$d/ $CT:OWL/cg10k_best/nc_v15/$d/ 2>/dev/null; done
  grep -q NC15_DONE $B/nc15.log && ssh $CT "grep -q CT_V15_NC_DONE OWL/cg10k_best/nc_v15/ct_worker.log" && break
  sleep 180
done; echo SYNC_V15_DONE
