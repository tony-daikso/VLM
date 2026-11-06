#!/bin/bash
# Exp N CPU chain part 2 (replaces the GS6 wait of nu_chain.sh): after the ath crops of this host -> stop the waiting
# nu_chain.sh -> focal-steatosis crops (all ids of this host) -> wait for gallstones/GS6_READY -> gallstones pass 2.
# 4 processes, nice 19.  usage: nu_chain2.sh TAG BASE prep1:mask1 [prep2:mask2 ...]
T=$1; B=$2; shift 2; cd "$B" || exit 1
PY=~/miniconda3/envs/radar/bin/python
while pgrep -f "[n]u_crop.py ath" > /dev/null; do sleep 20; done
pkill -f "[n]u_chain.sh $T"
mkdir -p focal_steatosis
nice -n 19 $PY scripts/nu_crop.py fst ids_n_${T}.txt focal_steatosis/crops 4 "$@" > focal_steatosis/crop_${T}.log 2>&1
while [ ! -f gallstones/GS6_READY ]; do sleep 60; done
GS6_SEG=$B/gallstones/gs6_seg nice -n 19 $PY scripts/nu_crop.py gs ids_n_${T}_gs_p2.txt gallstones/crops 4 "$@" > gallstones/crop_${T}.log 2>&1
echo CHAIN2_DONE $(date +%T)
