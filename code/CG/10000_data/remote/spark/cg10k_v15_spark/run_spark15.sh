#!/bin/bash
# Spark helper for CT-rate's cg10k 1.5 mm vertebra run (tmux cg10k_v15 on CT-rate). Resumable.
# usage: tmux new -d -s cg10k_v15_spark 'bash ~/OWL/cg10k_v15_spark/run_spark15.sh 2>&1 | tee -a ~/OWL/cg10k_v15_spark/logs/run.log'
# stop:  touch ~/OWL/cg10k_v15_spark/STOP   (running round finishes ~2-4 min, syncs, releases claims, exit)
# fast stop: kill -TERM <coord pid>          (workers killed, finished cases synced, the rest released)
# resume: rm STOP; rerun (leftover Spark claims on CT-rate are synced/released at start)
D=$HOME/OWL/cg10k_v15_spark
cd $D; mkdir -p logs state
[ -f STOP ] && { echo "$(date '+%F %T') STOP exists - remove it to run"; exit 1; }
N_W=${N_W:-3} THREADS=${THREADS:-4} ROUND=${ROUND:-60} $HOME/radar_env/bin/python -u spark_coord15.py < /dev/null
echo "$(date '+%F %T') coord exit rc=$?"
