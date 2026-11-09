#!/bin/bash
# A6000 thermal guard for pelvis15_ft (agent pelvis15_ft): every 30 s; GPU > 85 C -> SIGSTOP pv15_ft.py processes, resume at <= 78 C.
# does not touch the 200 W power cap. ends when no pv15_ft.py process is left for 10 min, or on touch .../pelvis15_ft/STOP_GUARD
B=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft; idle=0; paused=0
while [ ! -f $B/STOP_GUARD ]; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  pids=$(pgrep -f "[p]v15_ft.py")
  if [ -z "$pids" ]; then idle=$((idle+1)); [ $idle -ge 20 ] && break; else idle=0; fi
  if [ "$t" -gt 85 ] && [ $paused = 0 ] && [ -n "$pids" ]; then kill -STOP $pids; paused=1; echo "$(date '+%m-%d %H:%M:%S') PAUSE gpu=${t}C" >> $B/logs/thermal.log; fi
  if [ $paused = 1 ] && [ "$t" -le 78 ]; then [ -n "$pids" ] && kill -CONT $pids; paused=0; echo "$(date '+%m-%d %H:%M:%S') RESUME gpu=${t}C" >> $B/logs/thermal.log; fi
  sleep 30
done
[ $paused = 1 ] && pids=$(pgrep -f "[p]v15_ft.py") && [ -n "$pids" ] && kill -CONT $pids
echo "$(date '+%m-%d %H:%M:%S') guard exit" >> $B/logs/thermal.log
