#!/bin/bash
# pelvis15_ft tier lane on the CT-rate 3080 (agent pelvis15_ft, 11-08): bf16, VRAM <= 9 GB. stop: touch runs/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft
bash scripts/tier_run.sh ct 295 t295sm --side 0.5 --mir --amp bf16 --vram-gb 9 --workers 4
bash scripts/tier_run.sh ct 297 t297s --side 0.5 --amp bf16 --vram-gb 9 --workers 4
echo "$(date "+%m-%d %H:%M") LANE_CT_DONE" >> logs/tier.log
