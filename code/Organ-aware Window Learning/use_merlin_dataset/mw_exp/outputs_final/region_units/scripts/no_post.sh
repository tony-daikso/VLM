#!/bin/bash
# Exp O post-processing (Mac relay; copy of Exp N nu_post.sh): probe on the 4090 GPU (nice 19, VRAM cap / threads given; one
# probe at a time via a lock dir) -> copy probe out*/ + logs to A6000 -> score-level stack on A6000 CPU (nice 19, 4 threads).
# usage: no_post.sh UNIT CODE[,CODE2:OUTNAME2] VRAM_GB THREADS
U=$1; CS=$2; VR=$3; TH=$4
ST="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expO_stage"
A6D=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
G() { ssh -n -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
until mkdir "$ST/probe.lock" 2>/dev/null; do sleep 30; done
echo "$(date +%T) probe on 4090 ($U)"
IFS=, read -ra PAIRS <<< "$CS"
for p in "${PAIRS[@]}"; do C=${p%%:*}; O=out; [ "$p" != "$C" ] && O=${p#*:}
  for try in 1 2; do
    G "cd /root/OWL/expO && MW_SRC=/root/OWL/mw_exp/src NT=$TH OMP_NUM_THREADS=$TH nice -n 19 /root/L/venv/bin/python scripts/no_probe.py $C $U/emb probe_inputs/main_$C.npz probe_inputs/labels_$C.csv $U/$O --device cuda --vram-gb $VR > $U/logs/probe_$C.log 2>&1 < /dev/null; tail -5 $U/logs/probe_$C.log"
    G "grep -q RU_PROBE_DONE /root/OWL/expO/$U/logs/probe_$C.log" && break
    echo "probe $C failed (try $try); waiting for no extraction consumer, then retry with 10 GB"
    while G "pgrep -f '[n]o_extract.py' >/dev/null"; do sleep 30; done; VR=10
  done
done
rmdir "$ST/probe.lock"
mkdir -p "$ST/$U"
rsync -a -e "ssh -p 25966" --exclude emb --exclude inbox root@tw-07.access.glows.ai:/root/OWL/expO/$U/ "$ST/$U/"
rsync -a --exclude emb "$ST/$U/" A6000:$A6D/$U/
ssh -n A6000 "cd $A6D && OMP_NUM_THREADS=4 nice -n 19 ~/miniconda3/envs/radar/bin/python scripts/no_stack.py $U > $U/stack.log 2>&1; grep -v -i warn $U/stack.log | tail -16"
echo "POST_DONE $U $(date +%T)"
