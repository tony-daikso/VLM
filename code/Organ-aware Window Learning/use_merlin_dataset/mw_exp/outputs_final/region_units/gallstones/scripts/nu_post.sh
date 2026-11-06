#!/bin/bash
# Exp N post-processing (Mac relay; copy of Exp M ru_post.sh): wait for FEED_DONE of UNIT -> probe on the 4090 GPU
# (nice 19, VRAM cap / threads given) -> copy probe out/ + logs to A6000 -> optional pre-stack step on A6000/CT-rate
# (PRE env, e.g. HU scalars) -> stack on A6000 CPU (nice 19, 4 threads).
# usage: nu_post.sh UNIT CODE VRAM_GB THREADS "STACK_CMD (run in A6000 region_units, python = $PY)"
U=$1; C=$2; VR=$3; TH=$4; SC=$5
ST="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expN_stage"
A6D=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
G() { ssh -n -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
until grep -q FEED_DONE "$ST/feed_$U.log"; do sleep 60; done
echo "$(date +%T) feed done; probe on 4090"
G "cd /root/OWL/expN && MW_SRC=/root/OWL/mw_exp/src NT=$TH OMP_NUM_THREADS=$TH nice -n 19 /root/L/venv/bin/python scripts/nu_probe.py $C $U/emb probe_inputs/main_$C.npz probe_inputs/labels_$C.csv $U/out --device cuda --vram-gb $VR > $U/logs/probe.log 2>&1 < /dev/null; tail -6 $U/logs/probe.log"
mkdir -p "$ST/$U/out"
rsync -a -e "ssh -p 25966" root@tw-07.access.glows.ai:/root/OWL/expN/$U/out/ "$ST/$U/out/"
rsync -a -e "ssh -p 25966" root@tw-07.access.glows.ai:/root/OWL/expN/$U/logs/ "$ST/$U/logs_4090/"
rsync -a "$ST/$U/out/" A6000:$A6D/$U/out/; rsync -a "$ST/$U/logs_4090/" A6000:$A6D/$U/logs_4090/
[ -n "$PRE" ] && eval "$PRE"
ssh -n A6000 "cd $A6D && PY=~/miniconda3/envs/radar/bin/python && OMP_NUM_THREADS=4 nice -n 19 $SC > $U/stack.log 2>&1; grep -v -i warn $U/stack.log | tail -14"
echo "POST_DONE $U $(date +%T)"
