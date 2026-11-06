#!/bin/bash
# Mac relay: copy the GS6 1.5 mm GB masks (Spark ~/OWL/gs6/seg) to A6000 and CT-rate gallstones/gs6_seg every 5 min;
# when the Spark GS6 inbox has been empty for 2 checks (or gs6_seg.py is gone), final sync + touch GS6_READY on both.
ST="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expN_stage/gs6_seg"; mkdir -p "$ST"
A6=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/gallstones; CT=OWL/mw_exp/region_units_tmp/gallstones
e=0
while true; do
  rsync -a --exclude '*.tmp*' sparks:OWL/gs6/seg/ "$ST/"
  rsync -a "$ST/" A6000:$A6/gs6_seg/; rsync -a "$ST/" CT-rate:$CT/gs6_seg/
  ni=$(ssh sparks 'ls ~/OWL/gs6/inbox | grep -c npz$'); run=$(ssh sparks 'pgrep -f "[g]s6_seg.py" | wc -l')
  echo "$(date +%T) seg $(ls "$ST" | wc -l) inbox $ni running $run"
  if [ "$ni" = "0" ] || [ "$run" = "0" ]; then e=$((e+1)); else e=0; fi
  if [ $e -ge 2 ]; then
    sleep 30; rsync -a --exclude '*.tmp*' sparks:OWL/gs6/seg/ "$ST/"; rsync -a "$ST/" A6000:$A6/gs6_seg/; rsync -a "$ST/" CT-rate:$CT/gs6_seg/
    ssh A6000 "touch $A6/GS6_READY"; ssh CT-rate "touch $CT/GS6_READY"; echo "GS6_SYNC_DONE $(ls "$ST" | wc -l) $(date +%T)"; break
  fi
  sleep 300
done
