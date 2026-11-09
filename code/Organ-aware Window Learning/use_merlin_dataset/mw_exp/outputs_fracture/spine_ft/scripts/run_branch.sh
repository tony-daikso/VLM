#!/bin/bash
# spine_ft / pelvis_ft driver (A6000 GPU). usage: run_branch.sh spine|pelvis "name1=args1;name2=args2;..."
# 1) each config: 1 seed, 5-fold (sitewise folds) -> OOF for all train rows + val (fold-model mean). val is written but
#    NEVER used for selection.  2) pick best config by train OOF (tier rows, site label).  3) extra seed of the best config
#    (seed0=1) -> runs/final_<best>/ = mean of the 2 seeds (oof_train.csv, val_scores.csv, summary.json).
# stop: touch outputs_fracture/<branch>_ft/runs/STOP (current epoch finishes). resume: rerun (finished folds/configs skipped).
B=$1; CFGS=$2
MW=~/Desktop/Tony/VLM/OWL/mw_exp; D=$MW/outputs_fracture/${B}_ft; PY=~/miniconda3/envs/radar/bin/python
S=$MW/outputs_fracture/spine_ft
cd $D; mkdir -p runs logs
PACKS="$S/in/pack_a6.npz $S/in/pack_ct.npz"
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver.log; }
run(){ # out args...
  o=$1; shift
  [ -f runs/$o/summary.json ] && return
  log "$o start ($*)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY $S/scripts/frac_ft.py --branch $B --labels $S/in/labels.csv \
    --packs $PACKS --out runs/$o "$@" > logs/$o.log 2>&1
  log "$o rc=$? $(grep -E 'FT train OOF' logs/$o.log)"
}
log "driver start pid $$ cfgs=$CFGS"
IFS=';' read -ra CL <<< "$CFGS"
for c in "${CL[@]}"; do
  [ -f runs/STOP ] && { log STOP; exit 0; }
  n=${c%%=*}; read -ra args <<< "${c#*=}"
  run s0_$n --seeds 1 --seed0 0 "${args[@]}"
done
[ -f runs/STOP ] && { log STOP; exit 0; }
BEST=$($PY -c "
import json,os,sys
r={}
for c in sys.argv[1].split(';'):
    n=c.split('=')[0]; f=f'runs/s0_{n}/summary.json'
    if os.path.exists(f): r[n]=json.load(open(f))['oof']
print(max(r,key=r.get))" "$CFGS")
echo "$BEST" > runs/best_cfg.txt; log "selected by train OOF: $BEST"
read -ra args <<< "$(echo "$CFGS" | tr ';' '\n' | grep "^$BEST=" | cut -d= -f2-)"
[ -f runs/STOP ] && { log STOP; exit 0; }
run s1_$BEST --seeds 1 --seed0 1 "${args[@]}"
$PY - "$BEST" <<'EOF'
import sys, json, os, pandas as pd
b = sys.argv[1]; R = [f"runs/s{k}_{b}" for k in (0, 1) if os.path.exists(f"runs/s{k}_{b}/summary.json")]
os.makedirs(f"runs/final_{b}", exist_ok=True)
for f in ("oof_train.csv", "val_scores.csv"):
    ds = [pd.read_csv(f"{r}/{f}") for r in R]; o = ds[0].copy(); o["ft"] = sum(d["ft"] for d in ds) / len(ds)
    o.to_csv(f"runs/final_{b}/{f}", index=False)
json.dump({"seeds": R, "per_seed": {r: json.load(open(f"{r}/summary.json")) for r in R}}, open(f"runs/final_{b}/summary.json", "w"), indent=1)
print("final from", R)
EOF
log "final_$BEST written"; touch runs/FT_DONE; log FT_DONE
