#!/bin/bash
# Spark CPU: crop relayed CTs (buf/) into roi/ with pv15_crop.py watch mode (8 procs, nice 10); waits for the mask sync.
# Also pushes finished ROIs to CT-rate pelvis15/roi every 5 min. stop: touch ~/Desktop/Tony/OWL/pelvis15/STOP (+ buf/DONE)
cd ~/Desktop/Tony/OWL/pelvis15; PY=~/radar_env/bin/python
until [ -f MASKS_OK ]; do sleep 30; done
( while [ ! -f STOP_PUSH ]; do rsync -a --ignore-existing --exclude "*.tmp.npz" -e "ssh -c aes128-gcm@openssh.com -o BatchMode=yes" roi/ daikso@192.168.0.189:/media/daikso/Tony/VLM/data/CT/merlin/pelvis15/roi/ >> logs/push.log 2>&1; [ -f WATCH_DONE ] && break; sleep 300; done ) &
OMP_NUM_THREADS=1 nice -n 10 $PY scripts/pv15_crop.py buf . 8 masks_a masks_b >> logs/watch.log 2>&1
touch WATCH_DONE; wait
