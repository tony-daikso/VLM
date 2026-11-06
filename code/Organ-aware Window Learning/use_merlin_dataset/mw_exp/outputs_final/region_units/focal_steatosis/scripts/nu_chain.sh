#!/bin/bash
# Exp N CPU crop chain (one per host, 4 processes, nice 19): gallstones pass 1 (non-GS6 ids) -> atherosclerosis (all ids of
# this host) -> wait for gallstones/GS6_READY (GS6 1.5 mm GB masks synced into gallstones/gs6_seg) -> gallstones pass 2
# (the GS6 = 3 mm GB < 100 vox ids, centred/regions on the GS6 GB when available).
# usage: nu_chain.sh TAG BASE prep1:mask1 [prep2:mask2 ...]     (TAG = a6 | ct; ids_n_TAG*.txt in BASE)
T=$1; B=$2; shift 2; cd "$B" || exit 1
PY=~/miniconda3/envs/radar/bin/python
mkdir -p gallstones atherosclerosis
nice -n 19 $PY scripts/nu_crop.py gs ids_n_${T}_gs_p1.txt gallstones/crops 4 "$@" > gallstones/crop_${T}_p1.log 2>&1
nice -n 19 $PY scripts/nu_crop.py ath ids_n_${T}.txt atherosclerosis/crops 4 "$@" > atherosclerosis/crop_${T}.log 2>&1
while [ ! -f gallstones/GS6_READY ]; do sleep 60; done
GS6_SEG=$B/gallstones/gs6_seg nice -n 19 $PY scripts/nu_crop.py gs ids_n_${T}_gs_p2.txt gallstones/crops 4 "$@" > gallstones/crop_${T}.log 2>&1
echo CHAIN_DONE $(date +%T)
