#!/bin/bash
# pelvis15_ft integration chain (A6000 CPU; agent pelvis15_ft 11-08): wait for CT-rate SCORE_ALL_DONE -> pull seed runs ->
# combine -> final_ft variants on top of rp_both_aa (spine15_add + RV3@rib add + PV15@pel add) -> select by stacker train CV only.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15_ft; PY=~/miniconda3/envs/radar/bin/python; F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture
CT=daikso@192.168.0.189; R=/media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/integ.log; }
until ssh -n $CT "grep -q 'SCORE_ALL_DONE\|FAILED twice' $R/logs/full.log"; do [ -f runs/STOP ] && exit 0; sleep 180; done
ssh -n $CT "grep -q SCORE_ALL_DONE $R/logs/full.log" || { log "CT scoring failed"; exit 1; }
for s in 0 1; do mkdir -p ct_final/full_297s_s$s; scp -q $CT:$R/runs/full_297s_s$s/{oof_train.csv,val_scores.csv,summary.json} ct_final/full_297s_s$s/; done
$PY scripts/combine_full.py final ct_final/full_297s_s0 ct_final/full_297s_s1 > logs/combine.log 2>&1 || { log "combine failed"; exit 1; }
log "combined: $(grep -m3 '"oof_' logs/combine.log | tr -d '\n ')"
cd $F/final_ft
SP15="--add FT_sp15=$F/spine15_ft@vert"; RV3="--add RV3=$F/rib_v3/cand/lev28@rib"; PV="--add PV15=$F/pelvis15/final_m400@pel"; NEW=$F/pelvis15_ft/final
run(){ t=$1; shift; [ -f $t/summary.json ] && return; OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 nice -n 10 $PY $F/pelvis15_ft/scripts/final_ft_pf.py --tag $t "$@" > $t.log 2>&1; echo "$(date '+%m-%d %H:%M') $t rc=$?" >> $F/pelvis15_ft/logs/integ.log; }
run pf_add $SP15 $RV3 $PV --add P15FT=$NEW@pel
run pf_replpv $SP15 $RV3 $PV --pelvis $NEW
run pf_replpv15 $SP15 $RV3 --add P15FT=$NEW@pel
run pf_dropU $SP15 $RV3 $PV --add P15FT=$NEW@pel --drop U_org_pel,U_pel
$PY - <<'PY' >> $F/pelvis15_ft/logs/integ.log
import json
from pathlib import Path
FT = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/final_ft"
rows = [(t, " · ".join(json.load(open(FT / t / "summary.json"))["selected"]), json.load(open(FT / t / "summary.json"))["selected_train_cv"])
        for t in ["rp_both_aa", "pf_add", "pf_replpv", "pf_replpv15", "pf_dropU"] if (FT / t / "summary.json").exists()]
rows.sort(key=lambda r: -r[2])
for r in rows: print(f"{r[0]:14s} {r[1]:45s} {r[2]:.4f}")
json.dump({"by_train_cv": rows, "selected": rows[0][0]}, open(FT.parent / "pelvis15_ft/pf_selection.json", "w"), indent=1, ensure_ascii=False)
print("SELECTED", rows[0][0])
PY
log INTEG_DONE
