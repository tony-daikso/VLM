#!/bin/bash
# push90b A6000 GPU chain (memory: full a6 ~27 GB, tier ~12.5 GB, 47 GB card):
#  wait tier tw_w2 + tw_attn done -> full lane F1 (f_a6_s5); wait lane T1 done (tw_bias, tw_w2sep) -> tier lane T4 (tw_w2bias, tw_a6rep).
# The decision for the 2nd full run (f_a6_s10 or a tier-winning variant) is made by the agent after the tiers.  stop: touch push90b/STOP
F=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; B=$F/push90b; cd $B; PY=/home/advantech/miniconda3/envs/radar/bin/python
gone(){ ! pgrep -f "^$PY $B/scripts/rs_probe3.py .* A/runs/$1 " > /dev/null; }
until { [ -f A/runs/tw_w2/summary.json ] || gone tw_w2; } && { [ -f A/runs/tw_attn/summary.json ] || gone tw_attn; }; do [ -f STOP ] && exit 0; sleep 60; done
A6="--side-csv $F/rib_straight/in/rib_side.csv --seeds 5 --side 0.5 --lev 0.3 --ctx --blocks mean,max:0,1,2:base,bone --nbmode tile"
setsid nohup bash scripts/lane.sh a6k-F1 $B $PY $B/scripts/rs_probe2_f.py "f_a6_s5|prod/f_a6_s5|$F/rib_straight/in/labels.csv|$F/push90/A/feats_v2|$A6 --seed0 5" > /dev/null 2>&1 < /dev/null &
until grep -q "a6k-T1 lane done\|a6k-T1 .*FAILED" logs/push90b.log; do [ -f STOP ] && exit 0; sleep 60; done
L=$F/rib_straight/in/labels_tier_probe.csv; FE=$F/push90/A/feats_tier
TA="--side-csv $F/rib_straight/in/rib_side.csv --seeds 3 --side 0.5 --lev 0.3 --ctx --blocks mean,max:0,1,2:base,bone --nbmode tile"
bash scripts/lane.sh a6k-T4 $B $PY $B/scripts/rs_probe3.py "tw_w2bias|A/runs/tw_w2bias|$L|$FE|$TA --nbw 2 --ctxw bias" "tw_a6rep|A/runs/tw_a6rep|$L|$FE|$TA"
echo "$(date "+%m-%d %H:%M") chainA done" >> logs/push90b.log
