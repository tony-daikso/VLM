#!/bin/bash
# Mac relay for the rented L40S (root@tw-07.access.glows.ai:26747, claim tag l40s26747). Resumable.
# start: bash run_g.sh            (detached: nohup + caffeinate; log logs/coord.log, pid state/coord.pid)
# stop:  touch STOP               (no new claims; running v15 round / nc in-flight finish, sync, release, exit)
# fast:  kill -TERM $(cat state/coord.pid)   (L40S workers killed, finished cases synced, the rest released)
# resume: rm STOP; bash run_g.sh  (leftover l40s26747 claims are synced or released at start)
# env knobs: N_W THREADS ROUND PAR (v15), NC_W NC_TH QUEUE (nc), FRAC (main start point), DO_V15 DO_NC
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"; mkdir -p logs state
[ -f STOP ] && { echo "STOP exists - remove it to run"; exit 1; }
if [ -f state/coord.pid ] && kill -0 "$(cat state/coord.pid)" 2>/dev/null; then echo "already running pid $(cat state/coord.pid)"; exit 1; fi
nohup /usr/bin/python3 -u g_coord.py >> logs/coord.log 2>&1 < /dev/null &
pid=$!; echo $pid > state/coord.pid
nohup caffeinate -dimsu -w $pid >/dev/null 2>&1 &
echo "started pid $pid (+ caffeinate -w; log $D/logs/coord.log)"
