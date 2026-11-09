#!/bin/bash
# push90b: build candidate score dirs from finished runs (train-only stats; only train labels read by combine3).
# rib: a6 10-seed (f_a6 + f_a6_s5) side, w2 side, -> zmix with lev28 + rs15.  PE: t_vert / any targets (head+side).
# pelvis: only if a full ctx pelvis run exists (P/runs/pf_*).  skips any candidate whose runs are incomplete.
F=/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; B=$F/push90b; cd $B; PY=/home/advantech/miniconda3/envs/radar/bin/python
C="$PY scripts/combine3.py"; Z="$PY scripts/zmix.py"; RL=$F/rib_straight/in/labels.csv; PL=$B/B/pe_probe_labels_v.csv
ok(){ for r in "$@"; do for k in 0 1 2 3 4; do [ -f $r/score_fold$k.csv ] || return 1; done; done; }
LEV=$F/rib_v3/cand/lev28; RS=$F/rib_straight/cand/rs15
if ok prod/f_a6_s5; then $C $RL t_rib ft_side cand/a6x10_side $F/push90/prod/f_a6 prod/f_a6_s5; $Z cand/mix3_a6x10 $LEV $RS cand/a6x10_side; fi
for w in f_w2 f_w2sep f_w2bias; do if ok prod/$w; then $C $RL t_rib ft_side cand/${w#f_}_side prod/$w; $Z cand/mix3_${w#f_} $LEV $RS cand/${w#f_}_side
  $Z cand/mix4_${w#f_} $LEV $RS $F/push90/cand/a6_side cand/${w#f_}_side
  ok prod/f_a6_s5 && { $C $RL t_rib ft_side cand/a6x10${w#f_}_side $F/push90/prod/f_a6 prod/f_a6_s5 prod/$w; $Z cand/mix3_a6x10${w#f_} $LEV $RS cand/a6x10${w#f_}_side; }; fi; done
for t in tv:t_vert y:y; do n=${t%%:*}; tg=${t#*:}; R=""; for r in B/prod/pe_${n}_s0 B/prod/pe_${n}_s5 B/prod/pe_${n}_s10 B/prod/pe_${n}_s15 B/prod/pe_${n}_s20 B/prod/pe_${n}_s25 B/prod/pe_${n}_s30; do ok $r && R="$R $r"; done
  [ -n "$R" ] && for col in head side; do $C $PL $tg ft_$col cand/pe_${n}$(echo $R | wc -w)r_$col $R; done; done
for r in P/runs/pf_*; do [ -d $r ] || continue; done
R=""; for r in P/runs/pf_*; do ok $r && R="$R $r"; done; [ -n "$R" ] && $C $F/pelvis15/labels.csv t_pel ft_side cand/pel_ctx_side $R
ls cand
