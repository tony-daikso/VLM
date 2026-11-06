#!/bin/bash
# Wait for CT-rate run_nc.sh to finish (logs/nc_run.log ends with 'ALL DONE' and probs_nc.csv exists), copy
# probs_nc.csv + series_selection_nc.csv in place, run eval_owl_cg10k_nc.py (section F of owl_infer/report.md).
# usage: nohup ./finish_nc_eval.sh > .../owl_infer/finish_nc_eval.log 2>&1 &
cd /Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data
OUT=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/owl_infer
ts() { date '+%F %T'; }
echo "$(ts) waiting for CT-rate run_nc.sh ALL DONE"
until ssh -o ConnectTimeout=20 CT-rate 'tail -1 ~/OWL/cg10k_infer/logs/nc_run.log | grep -q "ALL DONE" && test -s /media/daikso/CTDATA/cg10k/owl/probs_nc.csv'; do sleep 600; done
scp -q CT-rate:/media/daikso/CTDATA/cg10k/owl/probs_nc.csv $OUT/probs_nc.csv || { echo "scp failed"; exit 1; }
echo "$(ts) probs_nc.csv: $(($(wc -l < $OUT/probs_nc.csv)-1)) rows"
python eval_owl_cg10k_nc.py && echo "$(ts) report section F updated"
