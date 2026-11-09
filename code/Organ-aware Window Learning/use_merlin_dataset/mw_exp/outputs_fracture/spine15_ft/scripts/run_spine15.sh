#!/bin/bash
# spine15_ft driver (A6000 GPU, fp32). Crops are made on CT-rate (frac15_raw) and pulled by sync_crops.sh.
# The A6000 GPU is shared (another agent's e292 FT ~19 GB + gs6 seg), so only ONE spine15 job runs here at a time;
# the second tune config (tf2ctx) runs on the CT-rate 3080 (run_ct.sh, bf16, accum 4) next to rib_ft2.
# 1) wait until every tierA id has a CT-rate crop status, final rsync of the tierA crops
# 2) s0_tf2lse here (seed 0, 5 sitewise folds, held-out TIER rows scored, fold models saved, no val)
# 3) wait for CT-rate s0_tf2ctx, pull its fold files/models; select by train OOF (tier rows, t_vert) -> runs/best_cfg.txt
# 4) seeds 1 and 2 of the best config here (fp32; concurrently if >= 26 GB GPU memory is free, else one after the other)
# 5) score_fill.sh (separate) scores val + non-tier train rows from the saved fold models -> runs/final_<best>
# stop: touch outputs_fracture/spine15_ft/runs/STOP (current epoch finishes); resume: rerun (finished folds/runs skipped)
MW=~/Desktop/Tony/VLM/OWL/mw_exp; S=$MW/outputs_fracture/spine15_ft; PY=~/miniconda3/envs/radar/bin/python
LAB=$MW/outputs_fracture/spine_ft/in/labels.csv; CT=daikso@192.168.0.189; RD=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw
cd $S; mkdir -p runs logs
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver.log; }
declare -A CF=([tf2lse]="--train-from 2 --pool lse" [tf2ctx]="--train-from 2 --pool lse --ctx")
log "driver start pid $$"
sed 's/.npz$//' tierA_files.txt | sort > .tierA_ids.txt
while true; do
  [ -f runs/STOP ] && { log STOP; exit 0; }
  nd=$(ssh -n $CT "cut -f1 $RD/status.tsv" | sort -u | comm -12 - .tierA_ids.txt | wc -l)
  [ "$nd" -ge "$(wc -l < .tierA_ids.txt)" ] && break
  sleep 120
done
rsync -a --ignore-missing-args --files-from=tierA_files.txt $CT:$RD/crops/ crops/; touch STOP_SYNC
log "tierA crops ready: $(ls crops | wc -l) files"
run(){ # out args...
  o=$1; shift
  [ -f runs/$o/summary.json ] && return
  log "$o start ($*)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels $LAB --crops crops --out runs/$o \
    --epochs 8 --vram-gb 19 --threads 3 --workers 2 --throttle 0 --no-val "$@" > logs/$o.log 2>&1
  log "$o rc=$? $(grep -E 'FT train OOF' logs/$o.log)"
}
run s0_tf2lse --seeds 1 --seed0 0 ${CF[tf2lse]}
[ -f runs/s0_tf2lse/summary.json ] || { log "s0_tf2lse failed -> driver exit"; exit 1; }
[ -f runs/STOP ] && { log STOP; exit 0; }
until ssh -n $CT "test -f $RD/runs/s0_tf2ctx/summary.json"; do [ -f runs/STOP ] && exit 0; sleep 120; done
rsync -a $CT:$RD/runs/s0_tf2ctx runs/ && log "pulled CT-rate s0_tf2ctx: $(grep -o '"oof": [0-9.]*' runs/s0_tf2ctx/summary.json)"
BEST=$($PY -c "
import json,os,sys
r={n: json.load(open(f'runs/s0_{n}/summary.json'))['oof'] for n in ('tf2lse','tf2ctx') if os.path.exists(f'runs/s0_{n}/summary.json')}
print(r, file=sys.stderr); print(max(r,key=r.get))" 2>> logs/driver.log)
echo "$BEST" > runs/best_cfg.txt; log "selected by train OOF (tier, t_vert): $BEST"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -ge 26000 ]; then
  run s1_$BEST --seeds 1 --seed0 1 ${CF[$BEST]} & sleep 90; run s2_$BEST --seeds 1 --seed0 2 ${CF[$BEST]} & wait
else
  run s1_$BEST --seeds 1 --seed0 1 ${CF[$BEST]}; [ -f runs/STOP ] && exit 0; run s2_$BEST --seeds 1 --seed0 2 ${CF[$BEST]}
fi
for k in 1 2; do [ -f runs/s${k}_$BEST/summary.json ] || { log "s${k}_$BEST failed -> driver exit"; exit 1; }; done
[ -f runs/STOP ] && { log STOP; exit 0; }
touch runs/TRAIN_DONE; log "TRAIN_DONE best=$BEST"
