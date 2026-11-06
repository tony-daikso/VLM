#!/bin/bash
# method-2 smoke test on Spark: 300 train / 100 val, 2 epochs, three configurations
cd ~/OWL/mw_ft/src
PY=~/radar_env/bin/python
C="--config ../config_sparks_ft.yaml --max-train 300 --max-val 100 --epochs 2 --standardize --std-cases 100 --results-subdir ft_smoke --warmup-steps 20"
$PY finetune.py $C --run smoke_frozen --train-stages 0 --no-train-first-conv > ../logs/smoke_frozen.log 2>&1
$PY finetune.py $C --run smoke_ft2 --train-stages 2 > ../logs/smoke_ft2.log 2>&1
$PY finetune.py $C --run smoke_ft2_trap --train-stages 2 --trap > ../logs/smoke_ft2_trap.log 2>&1
echo done > ../logs/SMOKE_DONE
