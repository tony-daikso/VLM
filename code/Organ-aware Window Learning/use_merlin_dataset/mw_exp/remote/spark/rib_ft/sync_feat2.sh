#!/bin/bash
# CT-rate feats2 (3080 part 2/3) -> Spark feats2. stop: touch ~/OWL/rib_ft/STOP_SYNC_F2
while [ ! -f ~/OWL/rib_ft/STOP_SYNC_F2 ]; do
  rsync -a --ignore-existing --exclude "*.tmp.npz" -e "ssh -c aes128-gcm@openssh.com -o BatchMode=yes" daikso@192.168.0.189:/media/daikso/Tony/VLM/data/CT/merlin/rib_ft/feats2/ ~/OWL/rib_ft/feats2/
  sleep 150
done
