#!/bin/bash
# push90b A6000 thermal guard: every 20 s; GPU temp > 85 C -> SIGSTOP all push90b python jobs; resume (SIGCONT) at <= 78 C.
# Also logs if the power limit is not 200 W (never changes it). Exits when push90b/STOP or push90b/STOP_GUARD exists.
B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90b; cd $B; paused=0
pids(){ pgrep -f "^/home/advantech/miniconda3/envs/radar/bin/python /home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90b/scripts/"; }
while [ ! -f STOP ] && [ ! -f STOP_GUARD ]; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
  [ "$pl" != "200" ] && echo "$(date "+%m-%d %H:%M") WARN power limit $pl" >> logs/guard.log
  if [ "$paused" = 0 ] && [ "$t" -gt 85 ]; then for p in $(pids); do kill -STOP $p; done; paused=1; echo "$(date "+%m-%d %H:%M") PAUSE temp $t" >> logs/guard.log; fi
  if [ "$paused" = 1 ] && [ "$t" -le 78 ]; then for p in $(pids); do kill -CONT $p; done; paused=0; echo "$(date "+%m-%d %H:%M") RESUME temp $t" >> logs/guard.log; fi
  echo "$(date "+%m-%d %H:%M:%S") temp $t paused $paused" > logs/guard.status; sleep 20
done
for p in $(pids); do kill -CONT $p; done
