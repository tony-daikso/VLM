#!/bin/bash
# spine15_ft phase 2 (A6000, replaces the tail of run_spine15.sh). Waits for s0_tf2lse (A6000) and s0_tf2ctx (CT-rate 3080),
# selects by train OOF (tier rows, t_vert) -> runs/best_cfg.txt, then: seed 2 on the A6000 (fp32, 200 W cap), seed 1 on the
# CT-rate 3080 (bf16; already queued speculatively for tf2ctx; started here for tf2lse), pulls the CT-rate runs, TRAIN_DONE.
# stop: touch runs/STOP ; resume: rerun (finished runs skipped; per-epoch ckpt)
MW=~/Desktop/Tony/VLM/OWL/mw_exp; S=$MW/outputs_fracture/spine15_ft; PY=~/miniconda3/envs/radar/bin/python
LAB=$MW/outputs_fracture/spine_ft/in/labels.csv; CT=daikso@192.168.0.189; RD=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw
cd $S; log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver.log; }
declare -A CF=([tf2lse]="--train-from 2 --pool lse" [tf2ctx]="--train-from 2 --pool lse --ctx")
run(){ o=$1; shift; [ -f runs/$o/summary.json ] && return
  log "$o start ($*)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels $LAB --crops crops --out runs/$o \
    --epochs 8 --vram-gb 19 --threads 3 --workers 2 --throttle 0 --no-val "$@" > logs/$o.log 2>&1
  log "$o rc=$? $(grep -E 'FT train OOF' logs/$o.log)"; }
log "phase2 start pid $$"
while pgrep -f "frac15_ft.py --labels .* --out runs/s0_tf2lse" > /dev/null; do sleep 60; done
run s0_tf2lse --seeds 1 --seed0 0 ${CF[tf2lse]}   # resumes from ckpt if the first process died
[ -f runs/s0_tf2lse/summary.json ] || { log "s0_tf2lse missing -> exit"; exit 1; }
until ssh -n $CT "test -f $RD/runs/s0_tf2ctx/summary.json"; do [ -f runs/STOP ] && exit 0; sleep 60; done
rsync -a --exclude 'ckpt_*' $CT:$RD/runs/s0_tf2ctx runs/
BEST=$($PY -c "
import json,sys
r={n: json.load(open(f'runs/s0_{n}/summary.json'))['oof'] for n in ('tf2lse','tf2ctx')}
print(r, file=sys.stderr); print(max(r,key=r.get))" 2>> logs/driver.log)
echo "$BEST" > runs/best_cfg.txt; log "selected by train OOF (tier, t_vert): $BEST"
if [ "$BEST" = tf2lse ]; then  # cancel the speculative tf2ctx seed 1 on the 3080, start tf2lse seed 1 there
  ssh -n $CT "touch $RD/runs/STOP; while pgrep -f '[o]ut runs/s1_tf2ctx' >/dev/null; do sleep 10; done; rm -f $RD/runs/STOP; cd $RD && (setsid nohup bash scripts/run_ct2.sh tf2lse 1 > /dev/null 2>&1 < /dev/null &)"
  log "3080: speculative s1_tf2ctx cancelled, s1_tf2lse started"
fi
run s2_$BEST --seeds 1 --seed0 2 ${CF[$BEST]}
[ -f runs/STOP ] && { log STOP; exit 0; }
until ssh -n $CT "test -f $RD/runs/s1_$BEST/summary.json"; do [ -f runs/STOP ] && exit 0; sleep 60; done
rsync -a --exclude 'ckpt_*' $CT:$RD/runs/s1_$BEST runs/ && log "pulled CT-rate s1_$BEST: $(grep -o '"oof": [0-9.]*' runs/s1_$BEST/summary.json)"
for k in 0 1 2; do [ -f runs/s${k}_$BEST/summary.json ] || { log "s${k}_$BEST missing -> exit"; exit 1; }; done
touch runs/TRAIN_DONE; log "TRAIN_DONE best=$BEST"
