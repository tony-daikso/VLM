#!/bin/bash
# Exp O CPU crop chain (one per host, 4 processes, nice 19): bd -> fa2 -> asc2 -> frp -> go, ids_n_TAG.txt (same host split
# as Exp N). CT-rate (shared disk): before each unit wait until the Exp O crops still on disk are < 12 GB (the Mac feeder
# deletes crops once their embeddings are back), so Exp O temp usage stays well under 30 GB.
# usage: no_chain.sh TAG BASE prep1:mask1 [prep2:mask2 ...]
T=$1; B=$2; shift 2; cd "$B" || exit 1
PY=~/miniconda3/envs/radar/bin/python
for p in bd:bile_duct_r2 fa2:free_air_r2 asc2:ascites_r2 frp:fracture_pelvis_r2 go:bowel_obstruction_r2; do
  C=${p%%:*}; U=${p##*:}; mkdir -p $U
  if [ "$T" = ct ]; then
    while [ "$(du -sm *_r2/crops 2>/dev/null | awk '{s+=$1} END {print s+0}')" -gt 12000 ]; do sleep 60; done
  fi
  [ -f STOP_O ] && break
  echo "$(date +%T) start $C"
  nice -n 19 $PY scripts/no_crop.py $C ids_n_${T}.txt $U/crops 4 "$@" > $U/crop_${T}.log 2>&1
  echo "$(date +%T) done $C $(tail -1 $U/crop_${T}.log)"
done
echo CHAIN_O_DONE $(date +%T)
