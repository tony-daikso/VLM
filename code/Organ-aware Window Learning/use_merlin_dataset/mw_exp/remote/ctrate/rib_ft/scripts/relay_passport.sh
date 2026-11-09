#!/bin/bash
# Mac relay only (no compute): stream raw Merlin CTs from the Passport to Spark ~/OWL/rib_ft/buf (Spark rib_prep.py --watch
# consumes + deletes them). Takes ids from the END of ids_order.txt (tier1 train, reversed) while CT-rate works forward.
# Skips ids whose roi exists on CT-rate or Spark (lists refreshed every 40 files). Buffer cap 24 files.
# stop: touch <this dir>/STOP_RELAY
D="$(cd "$(dirname "$0")" && pwd)"; SRC="/Volumes/My Passport/Tony/VLM/data/CT/merlin/merlin_data"
IDS="$D/../ids_order.txt"; DONE="$D/../relay_done.txt"; touch "$DONE"
refresh(){ { ssh CT-rate 'ls /media/daikso/Tony/VLM/data/CT/merlin/rib_ft/roi'; ssh sparks 'ls ~/OWL/rib_ft/roi'; cat "$DONE"; } 2>/dev/null | sed 's/\.npz$//' | sort -u > "$D/../.have.txt"; }
refresh; n=0
tail -r "$IDS" | while read cid; do
  [ -f "$D/STOP_RELAY" ] && break
  if [ $((n % 40)) -eq 0 ]; then refresh; fi
  grep -qx "$cid" "$D/../.have.txt" && continue
  # CT-rate's forward pointer reached this id? stop when CT-rate status already lists it
  while [ "$(ssh sparks 'ls ~/OWL/rib_ft/buf | grep -c nii.gz')" -ge 24 ]; do sleep 3; done
  cat "$SRC/$cid.nii.gz" | ssh -c aes128-gcm@openssh.com sparks "cat > ~/OWL/rib_ft/buf/.$cid.part && mv ~/OWL/rib_ft/buf/.$cid.part ~/OWL/rib_ft/buf/$cid.nii.gz" && echo "$cid" >> "$DONE"
  n=$((n+1)); [ $((n % 50)) -eq 0 ] && echo "$(date '+%H:%M') sent $n last $cid"
done
echo "$(date '+%H:%M') relay end n=$n"
