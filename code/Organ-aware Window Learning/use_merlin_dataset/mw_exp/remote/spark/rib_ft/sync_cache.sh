#!/bin/bash
# Spark cache -> CT-rate cache (3080 training). stop: touch ~/OWL/rib_ft/STOP_SYNC
while [ ! -f ~/OWL/rib_ft/STOP_SYNC ]; do
  rsync -a --ignore-existing --exclude "*.tmp.npz" --exclude STOP -e "ssh -c aes128-gcm@openssh.com -o BatchMode=yes" ~/OWL/rib_ft/cache/ daikso@192.168.0.189:/media/daikso/Tony/VLM/data/CT/merlin/rib_ft/cache/
  echo "$(date +%H:%M) synced $(ls ~/OWL/rib_ft/cache | wc -l)"; sleep 120
done
