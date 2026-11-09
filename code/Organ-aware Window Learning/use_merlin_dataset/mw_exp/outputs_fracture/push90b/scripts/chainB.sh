#!/bin/bash
# push90b chainB: after full F1 (f_a6_s5) finishes, run the full 5-seed (seeds 0-4) version of the best ctx tier variant
# (side > .892 required; candidates tw_w2 / tw_w2sep / tw_w2bias, tw_w2 = .8931 at launch).  memory: one full run at a time.
F=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; B=$F/push90b; cd $B; PY=/home/advantech/miniconda3/envs/radar/bin/python
until [ -f prod/f_a6_s5/summary.json ] || grep -q "a6k-F1 f_a6_s5 FAILED" logs/push90b.log; do [ -f STOP ] && exit 0; sleep 60; done
# wait for T1 (tw_w2sep) so its result is known; tw_w2bias only if already finished
until grep -q "a6k-T1 lane done\|a6k-T1 .*FAILED" logs/push90b.log; do [ -f STOP ] && exit 0; sleep 60; done
W=$($PY - <<'PYE'
import json, os
best, bv = None, 0.892
for n, x in [("tw_w2", "--nbw 2"), ("tw_w2sep", "--nbw 2 --nbsep"), ("tw_w2bias", "--nbw 2 --ctxw bias")]:
    f = f"A/runs/{n}/summary.json"
    if os.path.exists(f):
        v = json.load(open(f))["oof_t_rib_side"]
        if v > bv: best, bv = (n, x), v
print(f"{best[0]}|{best[1]}|{bv:.4f}" if best else "none")
PYE
)
echo "$(date "+%m-%d %H:%M") chainB pick: $W" >> logs/push90b.log
[ "$W" = none ] && { echo "$(date "+%m-%d %H:%M") chainB: no winner -> f_a6_s10" >> logs/push90b.log; SC=rs_probe2_f.py; N=f_a6_s10; X="--seed0 10"; } || { IFS="|" read -r T X V <<< "$W"; SC=rs_probe3.py; N=f_${T#tw_}; }
A6="--side-csv $F/rib_straight/in/rib_side.csv --seeds 5 --side 0.5 --lev 0.3 --ctx --blocks mean,max:0,1,2:base,bone --nbmode tile"
bash scripts/lane.sh a6k-F2 $B $PY $B/scripts/$SC "$N|prod/$N|$F/rib_straight/in/labels.csv|$F/push90/A/feats_v2|$A6 $X"
