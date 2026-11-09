#!/bin/bash
# pull CT-rate pelvis15 features f_* -> Spark every 5 min. stop: touch ~/Desktop/Tony/OWL/pelvis15/STOP_SYNC
cd ~/Desktop/Tony/OWL/pelvis15
while [ ! -f STOP_SYNC ]; do
  for d in $(ssh -o BatchMode=yes daikso@192.168.0.189 "cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15 && ls -d f_*"); do
    mkdir -p $d; rsync -a --ignore-existing --exclude "*.tmp.npz" -e "ssh -c aes128-gcm@openssh.com -o BatchMode=yes" daikso@192.168.0.189:/media/daikso/Tony/VLM/data/CT/merlin/pelvis15/$d/ $d/; done
  sleep 300
done
