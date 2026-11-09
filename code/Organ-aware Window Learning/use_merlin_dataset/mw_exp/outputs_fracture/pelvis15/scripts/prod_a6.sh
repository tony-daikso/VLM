#!/bin/bash
# pelvis15 production lane 3 (agent fracture_rp, 11-07 23:35; A6000 GPU, 200 W cap untouched): waits for the f_295 pull, load-checks, then
# Dataset295 stage0-1, side 0.5, maxinst 400, seeds 20-24. stop: touch outputs_fracture/pelvis15/STOP_PROD
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15; PY=~/miniconda3/envs/radar/bin/python
while pgrep -f "[r]sync -a --ignore-existing --exclude \*.tmp.npz --exclude FULL_DONE" >/dev/null; do sleep 20; done
n=$(ls f_295 | grep -c "^AC.*[0-9a-f]\.npz$"); echo "f_295 files $n" >> logs/prod_a6.out; [ "$n" -ge 19973 ] || { echo "INCOMPLETE" >> logs/prod_a6.out; exit 1; }
n=p295s_m400_s20; [ -f runs/$n/summary.json ] || [ -f STOP_PROD ] || OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_probe.py labels.csv f_295 runs/$n --seeds 5 --seed0 20 --side 0.5 --side-csv pel_side.csv --maxinst 400 > logs/$n.log 2>&1
grep PROBE logs/$n.log >> logs/prod_summary.txt
