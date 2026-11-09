#!/bin/bash
# rib_ft tuning driver. usage: run_tune.sh HOSTTAG PY BASE "cfg1 cfg2 ..." ; cfg = name:args (args with , for spaces)
# waits until all tier0 trainable train ids are cached, then runs each config (5 folds, --no-val, train OOF only).
# stop: touch BASE/runs/STOP (current epoch finishes -> exit 3). resume: rerun (finished folds are skipped).
TAG=$1; PY=$2; B=$3; CFGS=$4; cd $B; mkdir -p runs logs
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver_$TAG.log; }
need=$($PY -c "import pandas as pd;d=pd.read_csv('labels.csv');print(' '.join(d[(d.split=='train')&(d.tier==0)&d.t_rib.notna()].study_id))")
while true; do miss=0; for c in $need; do [ -f cache/$c.npz ] || miss=$((miss+1)); done; [ $miss -lt 15 ] && break; log "waiting cache, missing $miss"; sleep 120; done
for cfg in $CFGS; do
  name=${cfg%%:*}; args=$(echo ${cfg#*:} | tr ',' ' ')
  [ -f runs/$name/oof_fold4.csv ] && [ -f runs/$name/oof_fold0.csv ] && continue
  [ -f runs/STOP ] && { log STOP; exit 0; }
  log "start $name ($args)"
  nice -n 5 $PY scripts/rib_ft.py labels.csv roi runs/$name --cache cache --no-val --stop runs/STOP $args > logs/$name.log 2>&1
  log "end $name rc=$? $(grep -c 'OOF AUROC' logs/$name.log) folds"
done
log "driver done"
