#!/bin/bash
# rib_v3 ablation (Spark GPU): full train 5-fold OOF, 3 seeds each, <=3 concurrent. stop: touch ~/Desktop/Tony/OWL/rib_v3/STOP
cd ~/Desktop/Tony/OWL/rib_v3; PY=~/radar_env/bin/python
run(){ n=$1; shift; [ -f runs/$n/summary.json ] && return; [ -f STOP ] && return
  while pgrep -f "runs/$n --" >/dev/null; do sleep 30; done; [ -f runs/$n/summary.json ] && { grep PROBE logs/$n.log >> logs/abl_summary.txt; return; }
  $PY scripts/rib_probe3.py labels.csv feats3 runs/$n --side-csv rib_side.csv --seeds 3 "$@" > logs/$n.log 2>&1; grep PROBE logs/$n.log >> logs/abl_summary.txt; }
( run v3_base; run v3_side --side 0.5; run v3_ci_side_lev --ctxin --side 0.5 --lev 0.3 ) &
( run v3_ci --ctxin; run v3_ci_side_ema --ctxin --side 0.5 --ema 0.995 ) &
( run v3_ci_side --ctxin --side 0.5; run v3_side_lev --side 0.5 --lev 0.3 ) &
wait; echo ABL_DONE >> logs/abl_summary.txt
