#!/bin/bash
# Mac orchestrator: runs all gs15 GPU-heavy post-processing on the rented 4090 (A6000 GPU reserved for other work).
# Only derived data goes to the 4090: crop embeddings, a CNN input pack (64^3 HU + GB mask), main_gs.npz (train/val, no test),
# and train/val label csvs (labels_gs subsets; never test). Nothing secret is printed.
#  phase tune : when >= 2,400 tier0 TRAIN crops exist (or tier0 requested): pack them -> 4090 CNN on train rows only
#               (no val rows at all), 3 configs, 1 seed, GS1 5-fold OOF -> best_args (highest train OOF).
#  phase tier0/full (when A6000 writes REQ_<mode>): emb -> 4090 probe (U/U_org/U_gb3) -> out[_tier0]/ + PROBE_DONE on A6000;
#               pack -> 4090 CNN (best_args, 2 seeds) -> cnn_tier0|cnn_full on A6000 -> A6000 (CPU) re-runs gs15_stack.py after
#               the driver's stack -> report.md updated; flag CNN_<mode>_DONE.
# Usage: nohup caffeinate -i bash gs15_remote.sh > gs15_stage/remote.log 2>&1 &     Stop: touch gs15_stage/STOP_REMOTE
ST=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/gs15_stage; L=$ST/gpu; mkdir -p $L/emb $L/in $L/res
AR=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones15; A6="ssh -o BatchMode=yes A6000"; APY='~/miniconda3/envs/radar/bin/python'
GP="-p 23688"; GH=glows@tw-07.access.glows.ai; G="ssh -o BatchMode=yes $GP $GH"; GR=OWL/gs15_gpu; GPY='~/L/venv/bin/python'
RS4="rsync -a -e 'ssh -p 23688'"
stop() { [ -f $ST/STOP_REMOTE ] || $A6 "test -f $AR/STOP"; }
log() { echo "$(date +%H:%M) $*"; }
$G "mkdir -p $GR/scripts $GR/emb $GR/in $GR/res"
rsync -a -e "ssh $GP" $ST/scripts_gpu/ $GH:$GR/scripts/
if ! $G "test -f $GR/in/main_gs.npz"; then
  rsync -a A6000:Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/probe_inputs/main_gs.npz $L/in/ && rsync -a -e "ssh $GP" $L/in/main_gs.npz $GH:$GR/in/
fi
cnn_run () {  # $1 tag, $2 labels file (local), $3 pack (local), $4 extra args ; result -> $L/res/$1
  rsync -a -e "ssh $GP" $2 $3 $GH:$GR/in/
  $G "cd $GR && nice -n 5 $GPY scripts/gs15_cnn_train.py . in/$(basename $2) res/$1 --pack in/$(basename $3) --vram-gb 10 $4 > res/$1.log 2>&1"
  mkdir -p $L/res; rsync -a -e "ssh $GP" $GH:$GR/res/$1 $GH:$GR/res/$1.log $L/res/
}
# ---------------- phase tune ----------------
if [ ! -f $L/best_args.txt ]; then
  while ! stop; do
    n=$($A6 "cd $AR && $APY - <<'PY'
import pandas as pd, os
ids = pd.read_csv('inputs/ids_prio.csv'); t0 = set(ids[ids.tier == 0].study_id)
l = pd.read_csv('inputs/labels_gs.csv'); assert set(l.split) <= {'train', 'val'}
l = l[(l.split == 'train') & l.study_id.isin(t0) & l.y.notna()]
l = l[[os.path.exists(f'crops/{s}.npz') for s in l.study_id]]
os.makedirs('cnn_tune', exist_ok=True); l.to_csv('cnn_tune/labels_tune.csv', index=False); print(len(l))
PY")
    log "tier0 train crops $n"; { [ "${n:-0}" -ge 2400 ] || $A6 "test -f $AR/REQ_tier0"; } && break; sleep 300
  done
  if ! stop; then
    $A6 "cd $AR && nice -n 10 $APY scripts/gs15_cnn_pack.py . cnn_tune/labels_tune.csv cnn_tune/pack_tune.npz"
    rsync -a A6000:$AR/cnn_tune/labels_tune.csv A6000:$AR/cnn_tune/pack_tune.npz $L/in/
    for k in "e14_lr1e-3:--epochs 14 --lr 1e-3" "e24_lr5e-4:--epochs 24 --lr 5e-4" "e10_lr2e-3:--epochs 10 --lr 2e-3"; do
      stop && break; t=${k%%:*}; args=${k#*:}
      cnn_run tune_$t $L/in/labels_tune.csv $L/in/pack_tune.npz "$args --seeds 1 --train-only"; log "tune $t $(grep 'CNN train OOF' $L/res/tune_$t.log)"
    done
    python3 - "$L" <<'PY'
import json, os, sys
L = sys.argv[1]; cfg = {"e14_lr1e-3": "--epochs 14 --lr 1e-3", "e24_lr5e-4": "--epochs 24 --lr 5e-4", "e10_lr2e-3": "--epochs 10 --lr 2e-3"}
r = {k: json.load(open(f"{L}/res/tune_{k}/summary.json"))["oof"] for k in cfg if os.path.exists(f"{L}/res/tune_{k}/summary.json")}
if r:
    b = max(r, key=r.get); open(f"{L}/best_args.txt", "w").write(cfg[b]); print("TUNE", r, "best", b, flush=True)
PY
    rsync -a $L/res/tune_* $L/best_args.txt A6000:$AR/cnn_tune/ 2>/dev/null
  fi
fi
BEST=$(cat $L/best_args.txt 2>/dev/null || echo "--epochs 14 --lr 1e-3")
# ---------------- phases tier0 / full ----------------
for MODE in tier0 full; do
  if [ $MODE = tier0 ]; then OUT=out_tier0; CD=cnn_tier0; TAG=stack_tier0; else OUT=out; CD=cnn_full; TAG=stack; fi
  $A6 "test -f $AR/CNN_${MODE}_DONE" && continue
  until $A6 "test -f $AR/REQ_$MODE"; do stop && exit 0; sleep 120; done
  log "REQ_$MODE seen"; rsync -a A6000:$AR/inputs/labels_$MODE.csv $L/in/
  if ! $A6 "test -f $AR/$OUT/PROBE_DONE"; then
    rsync -a A6000:$AR/emb/ $L/emb/ && rsync -a -e "ssh $GP" $L/emb/ $GH:$GR/emb/ && rsync -a -e "ssh $GP" $L/in/labels_$MODE.csv $GH:$GR/in/
    $G "cd $GR && MW_SRC=scripts NT=4 nice -n 5 $GPY scripts/gs15_probe.py gs15 emb in/main_gs.npz in/labels_$MODE.csv res/$OUT --device cuda --vram-gb 6 --variants U,U_org,U_gb3 > res/probe_$MODE.log 2>&1"
    rsync -a -e "ssh $GP" $GH:$GR/res/$OUT $GH:$GR/res/probe_$MODE.log $L/res/
    if grep -q RU_PROBE_DONE $L/res/probe_$MODE.log; then
      $A6 "mkdir -p $AR/$OUT"; rsync -a $L/res/$OUT/ A6000:$AR/$OUT/; rsync -a $L/res/probe_$MODE.log A6000:$AR/logs/probe_${MODE}_4090.log
      $A6 "touch $AR/$OUT/PROBE_DONE"; log "probe $MODE done: $(grep 'train OOF' $L/res/probe_$MODE.log | tr '\n' ' ')"
    else log "probe $MODE FAILED on 4090 (A6000 falls back to CPU after 180 min)"; fi
  fi
  stop && exit 0
  $A6 "cd $AR && nice -n 10 $APY scripts/gs15_cnn_pack.py . inputs/labels_$MODE.csv inputs/pack_$MODE.npz" && rsync -a A6000:$AR/inputs/pack_$MODE.npz $L/in/
  cnn_run $CD $L/in/labels_$MODE.csv $L/in/pack_$MODE.npz "$BEST --seeds 2"
  if grep -q GS15_CNN_DONE $L/res/$CD.log; then
    rsync -a $L/res/$CD A6000:$AR/ && rsync -a $L/res/$CD.log A6000:$AR/logs/cnn_${MODE}_4090.log; log "cnn $MODE: $(grep -E 'CNN (train OOF|val)' $L/res/$CD.log | tr '\n' ' ')"
    $A6 "cd $AR && for i in \$(seq 1 300); do grep -q GS15_STACK_DONE logs/$TAG.log 2>/dev/null && break; sleep 60; done; nice -n 5 $APY scripts/gs15_stack.py ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final ~/$AR $MODE > logs/${TAG}_cnn.log 2>&1 && touch CNN_${MODE}_DONE"
    log "stack with cnn $MODE: $($A6 "grep SELECTED $AR/logs/${TAG}_cnn.log")"
  else log "cnn $MODE FAILED"; $A6 "touch $AR/CNN_${MODE}_FAILED"; fi
done
log "REMOTE_DONE"
