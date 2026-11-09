#!/bin/bash
# push90: pull feature files every 5 min (A6000 side): Spark + CT-rate rib v2 feats -> A/feats_v2 ; CT-rate PE feats -> B/feats.
# skips *.tmp.npz; stops after 10 h or when push90/STOP_SYNC exists. rsync failures are logged only.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90; CT=daikso@192.168.0.189; SP=daikso_spark@192.168.0.95
end=$(( $(date +%s) + 36000 ))
while [ ! -f STOP_SYNC ] && [ $(date +%s) -lt $end ]; do
  rsync -a --ignore-existing --exclude "*.tmp.npz" $SP:Desktop/Tony/OWL/push90/A/feats_v2/ A/feats_v2/ 2>> logs/sync.err
  rsync -a --ignore-existing --exclude "*.tmp.npz" $CT:/media/daikso/Tony/VLM/data/CT/merlin/push90/A/feats_v2/ A/feats_v2/ 2>> logs/sync.err
  rsync -a --ignore-existing --exclude "*.tmp.npz" $CT:/media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/feats/ B/feats/ 2>> logs/sync.err
  echo "$(date "+%H:%M") v2 $(ls A/feats_v2 | wc -l) pe $(ls B/feats | wc -l)" > logs/sync.status
  sleep 300
done
