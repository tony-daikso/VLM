#!/bin/bash
# pelvis15 encoder comparison (CT-rate 3080, <=6 GB): tier ids (train pelvis+ 309 + 1,500 negatives) with frozen
# Dataset292 / 294 / 295 stage 0-1 (baseline+bone). stop: touch .../pelvis15/STOP_FT
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15; PY=~/miniconda3/envs/radar/bin/python
T292=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset292*/nnUNet*); T295=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset295*/nnUNet*)
T294=$(ls -d /media/daikso/Tony/VLM/models/ts/Dataset294*/nnUNet*)
while [ ! -f STOP_FT ]; do
  if pgrep -f "frac15_ft.py" >/dev/null; then sleep 60; continue; fi   # 3080 belongs to spine15 scoring while it runs
  for e in 292 294 295; do eval T=\$T$e; mkdir -p f_$e
    $PY scripts/pv15_feat.py roi f_$e --ts $T --bone --once --vram-gb 6 --ids ids_tier.txt --only-ids >> logs/feat_tier_$e.log 2>&1; done
  n=$(comm -12 <(sort ids_tier.txt) <(ls f_295 | sed 's/\.npz$//' | sort) | wc -l)
  echo "$(date '+%H:%M') tier feats $n" >> logs/feat_tier.log
  [ $n -ge $(($(wc -l < ids_tier.txt) - 30)) ] && grep -q "CROP\|start N=12" logs/driver.log && [ $(comm -23 <(sort ids_tier.txt) <(ls roi | sed 's/\.npz$//' | sort) | wc -l) -le 30 ] && break
  sleep 60
done
touch TIER_FEAT_DONE
