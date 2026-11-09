#!/bin/bash
# CT-rate: wait until every tier tile has arrived (Spark push) and the 3080 has >= 5 GB free, then run tier B (Dataset295).
B=/media/daikso/Tony/VLM/data/CT/merlin/rib_straight; cd $B
for i in $(seq 1 240); do
  [ -f runs/STOP ] && exit 0
  miss=$(while read c; do [ -f tiles/$c.npz ] || echo $c; done < in/ids_tier.txt | wc -l)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$miss" = 0 ] && [ "$free" -ge 5000 ]; then break; fi
  sleep 60
done
[ "$miss" = 0 ] || { echo "$(date '+%m-%d %H:%M') tier tiles still missing ($miss) -> give up" >> logs/tier.log; exit 1; }
sleep 120   # let the tile rsync finish writing the last file
bash scripts/rs_tier.sh ct 295 t295sl --side .5 --lev .3
