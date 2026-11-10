#!/bin/bash
# score CG main with every saved full-train probe run -> runs/<leaf>/<run>/cg_scores.csv. usage: score_cg.sh GPU LEAF...
cd ~/Tony; . venv/bin/activate; g=$1; shift; export CUDA_VISIBLE_DEVICES=$g
declare -A FD=([lev28]=cg/out/feats3 [ftrib]=cg/out/feats3 [rs15]=cg/out/feats [a6]=cg/out/feats_v2 [pe]=cg/pe_feats [pe2]=cg/pe_feats)
for leaf in "$@"; do for r in runs/$leaf/*/; do r=${r%/}; [ -f $r/cg_scores.csv ] && continue
  eng=fullfit_rs.py; [ $leaf = ftrib ] && eng=fullfit_rp2.py
  python $eng x ${FD[$leaf]} $r --mode score --score-ids cg/ids_main.txt --score-out $r/cg_scores.csv >> logs/score_cg_$leaf.log 2>&1
  echo "$(date "+%m-%d %H:%M") $r rc=$? $(tail -1 logs/score_cg_$leaf.log)" >> logs/score_cg.log
done; done
