#!/bin/bash
# Exp M post-processing (Mac relay): wait for FEED_DONE of UNIT -> probe on the 4090 GPU (VRAM cap 10 GB, 4 threads,
# nice 19) -> copy probe out/ to A6000 -> stack on A6000 CPU (nice 19).  usage: ru_post.sh UNIT CODE
U=$1; C=$2
ST="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expM_stage"
A6D=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
G() { ssh -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
until grep -q FEED_DONE "$ST/feed_$U.log"; do sleep 60; done
echo "$(date +%T) feed done; probe on 4090"
G "cd /root/OWL/expM && MW_SRC=/root/OWL/mw_exp/src NT=4 OMP_NUM_THREADS=4 nice -n 19 /root/L/venv/bin/python scripts/ru_probe.py $C $U/emb probe_inputs/main_$C.npz probe_inputs/labels_$C.csv $U/out --device cuda --vram-gb 10 > $U/logs/probe.log 2>&1 < /dev/null; tail -6 $U/logs/probe.log"
mkdir -p "$ST/$U/out"
rsync -a -e "ssh -p 25966" root@tw-07.access.glows.ai:/root/OWL/expM/$U/out/ "$ST/$U/out/"
rsync -a -e "ssh -p 25966" root@tw-07.access.glows.ai:/root/OWL/expM/$U/logs/ "$ST/$U/logs_4090/"
rsync -a "$ST/$U/out/" A6000:$A6D/$U/out/; rsync -a "$ST/$U/logs_4090/" A6000:$A6D/$U/logs_4090/
ssh A6000 "cd $A6D && OMP_NUM_THREADS=4 nice -n 19 ~/miniconda3/envs/radar/bin/python scripts/ru_stack.py ibroot $U $U > $U/stack.log 2>&1; grep -v -i warn $U/stack.log | tail -12"
echo "POST_DONE $U $(date +%T)"
