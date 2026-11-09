#!/bin/bash
# pull Exp K rib CPRs from Spark every 3 min (stop: touch STOP_SYNC)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/ribs_hr
while [ ! -f STOP_SYNC ]; do
  rsync -a --exclude "*.tmp.npz" daikso_spark@192.168.0.95:OWL/ribs_hr/out/cpr/ cpr/
  echo "$(date +%T) $(ls cpr | wc -l) $(df -h / | tail -1 | awk "{print \$4}")" >> sync.log
  sleep 180
done
