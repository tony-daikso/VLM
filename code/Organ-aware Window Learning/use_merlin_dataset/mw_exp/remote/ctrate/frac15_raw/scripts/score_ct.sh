#!/bin/bash
# spine15_ft wrap-up scoring on the CT-rate 3080 (all crops local). usage: score_ct.sh BEST   (runs/s0_BEST holds the fold models)
# waits until every labels id has a crop status (back-fills status for crops written without one), stops the crop workers,
# then scores val + non-tier train rows with the 5 fold models (same precision as training) -> runs/s0_BEST/{oof_train,val_scores,summary}
# -> spine15_final/ ; touch runs/SCORE_DONE. stop: touch runs/STOP
R=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw; cd $R; PY=~/miniconda3/envs/radar/bin/python; B=$1
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/run_ct.log; }
log "score_ct start best=$B"
cut -d, -f1 labels.csv | tail -n +2 | sort > .all_ids.txt
while true; do
  [ -f runs/STOP ] && exit 0
  comm -23 <(ls crops | grep -v tmp | sed 's/.npz$//' | sort) <(cut -f1 status.tsv | sort -u) > .nostat.txt
  if [ -s .nostat.txt ]; then $PY -c "
import numpy as np, sys
for c in open('.nostat.txt').read().split():
    try: z = np.load(f'crops/{c}.npz'); p = z['present']; print(f'{c}\tok n{int(p.sum())} s15_{int((z[\"src\"] == 1).sum())} (status backfill)\t0')
    except Exception: pass" >> status.tsv; fi
  n=$(cut -f1 status.tsv | sort -u | comm -12 - .all_ids.txt | wc -l)
  [ "$n" -ge "$(wc -l < .all_ids.txt)" ] && break
  sleep 120
done
touch STOP_BUF; log "all crops have a status; crop workers stopped (STOP_BUF; FUSE pool finished its list)"
X=$($PY -c "import json; a=json.load(open('runs/s0_$B/summary.json'))['args']; print('--train-from',a['train_from'],'--pool',a['pool'],'--mask',a['mask'],'--wins',a['wins'],'--amp',a['amp'],'--ctx' if a['ctx'] else '')")
log "scoring runs/s0_$B ($X)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels labels.csv --crops crops --out runs/s0_$B \
  --src ~/OWL/mw_exp/src --score --seeds 1 --seed0 0 --vram-gb 6 --pred-bs 2 --threads 3 --workers 3 $X > logs/score_s0_$B.log 2>&1
rc=$?; log "score rc=$rc $(grep 'FT train OOF' logs/score_s0_$B.log)"
[ $rc = 0 ] || exit 1
mkdir -p spine15_final; cp runs/s0_$B/oof_train.csv runs/s0_$B/val_scores.csv runs/s0_$B/summary.json spine15_final/
touch runs/SCORE_DONE; log SCORE_DONE
