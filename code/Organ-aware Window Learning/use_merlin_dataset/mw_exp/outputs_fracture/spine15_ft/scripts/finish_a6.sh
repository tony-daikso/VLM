#!/bin/bash
# spine15_ft wrap-up (A6000; coordinator 14:45: no extra seeds). 1) wait for s0_tf2lse (GPU) to finish, 2) pull CT-rate
# s0_tf2ctx summary, select by train OOF (tier, t_vert) -> runs/best_cfg.txt, 3) push the selected run to CT-rate and start
# score_ct.sh there (3080; A6000 GPU is released here), 4) wait for SCORE_DONE, pull spine15_final -> spine15_ft/, then CPU:
# spine15_report.py + final_ft integration (with spine15 head + rib_ft) + compare vs main (.872) / with_rib (.878).
MW=~/Desktop/Tony/VLM/OWL/mw_exp; S=$MW/outputs_fracture/spine15_ft; F=$MW/outputs_fracture; PY=~/miniconda3/envs/radar/bin/python
CT=daikso@192.168.0.189; RD=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw
cd $S; log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver.log; }
log "finish_a6 start pid $$"
while pgrep -f "[f]rac15_ft.py --labels .* --out runs/s0_tf2lse" > /dev/null; do sleep 60; done
[ -f runs/s0_tf2lse/summary.json ] || { log "s0_tf2lse has no summary -> exit"; exit 1; }
log "s0_tf2lse: $(grep -o '"oof": [0-9.]*' runs/s0_tf2lse/summary.json); A6000 GPU released"
rsync -a --exclude 'ckpt_*' $CT:$RD/runs/s0_tf2ctx runs/
BEST=$($PY -c "
import json,sys
r={n: json.load(open(f'runs/s0_{n}/summary.json'))['oof'] for n in ('tf2lse','tf2ctx')}
print(r, file=sys.stderr); print(max(r,key=r.get))" 2>> logs/driver.log)
echo "$BEST" > runs/best_cfg.txt; log "selected by train OOF (tier, t_vert): $BEST"
[ "$BEST" = tf2lse ] && rsync -a --exclude 'ckpt_*' runs/s0_tf2lse $CT:$RD/runs/
ssh -n $CT "cd $RD && (setsid nohup bash scripts/score_ct.sh $BEST > /dev/null 2>&1 < /dev/null &)"; log "score_ct.sh $BEST started on CT-rate"
until ssh -n $CT "test -f $RD/runs/SCORE_DONE"; do sleep 120; done
rsync -a $CT:$RD/spine15_final/ ./ && log "pulled spine15_final -> spine15_ft/{oof_train,val_scores,summary}"
mkdir -p runs/final_$BEST; cp oof_train.csv val_scores.csv summary.json runs/final_$BEST/
export OMP_NUM_THREADS=4
nice -n 10 $PY scripts/spine15_report.py > logs/report.log 2>&1; log "report rc=$?"
cd $F/final_ft
nice -n 10 $PY scripts/final_ft.py --tag spine15_add --rib $F/rib_ft --add FT_sp15=$S@vert > spine15_add.log 2>&1; log "final_ft spine15_add rc=$?"
nice -n 10 $PY scripts/final_ft.py --tag spine15_repl --rib $F/rib_ft --spine $S > spine15_repl.log 2>&1; log "final_ft spine15_repl rc=$?"
touch $S/runs/INTEGRATION_DONE; log INTEGRATION_DONE
