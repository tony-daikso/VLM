#!/bin/bash
# A6000 -> CT-rate: every 2 min copy finished fold files + fold models (model before fold file; no ckpt) into the CT-rate run dirs
# for --score filling; ends after the A6000 full lane finished and a last sync. agent pelvis15_ft 11-08. stop: touch runs/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft; CT=daikso@192.168.0.189; R=/media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft
sync1(){ for s in 0 1; do d=runs/full_297s_s$s; [ -d $d ] || continue
  for f in $d/fold?.npz; do [ -f "$f" ] || continue; k=$(basename $f .npz); k=${k#fold}; m=$d/model_f${k}_s$s.pt; [ -f $m ] || continue
    if ! ssh -n $CT "test -f $R/$f"; then scp -q $m $CT:$R/$m.tmp && ssh -n $CT "mv $R/$m.tmp $R/$m" && scp -q $f $CT:$R/$f.tmp.npz && ssh -n $CT "mv $R/$f.tmp.npz $R/$f" \
      && echo "$(date '+%m-%d %H:%M') pushed $f + model" >> logs/full.log; fi; done; done; }
ssh -n $CT "mkdir -p $R/runs/full_297s_s0 $R/runs/full_297s_s1"
while true; do
  [ -f runs/STOP ] && exit 0
  sync1
  grep -q "a6 lane done\|a6 job .* FAILED" logs/full.log && { sync1; break; }
  sleep 120
done
echo "$(date '+%m-%d %H:%M') PUSH_CT_DONE" >> logs/full.log
