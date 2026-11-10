#!/bin/bash
# cg10k_nc sandbox: same refit scripts as cg10k_best (paths cg10k_best -> cg10k_nc), the "delay" slot of every script
# = the nc set (ids prefixed D_ inside the refits, renamed to nc at the end). main-set INPUTS (embeddings, unit
# embeddings, feature csvs, 1.5 mm embeddings) are reused read-only from cg10k_best (symlinked dirs / copied small files);
# all refits / stacks / scores are recomputed here with main + nc rows -> C+ and C- scores from the same model runs.
set -e
B0=~/Desktop/Tony/VLM/OWL/cg10k_best; B=~/Desktop/Tony/VLM/OWL/cg10k_nc
cd $B
for f in $B0/*.py; do sed 's#cg10k_best#cg10k_nc#g' $f > $B/$(basename $f); done
mkdir -p mat feat/main feat/delay e3late bemb frac osteo_v15/data gs6 best/main best/delay
ln -sfn $B0/mat/cg_main.npz mat/cg_main.npz
cp -n $B0/feat/main/* feat/main/
cp -n $B0/e3late/probs_main.csv e3late/
for u in hn fa asc ath rlq bd5 frp gs5 gs; do mkdir -p bemb/$u
  ln -sfn $B0/bemb/$u/main bemb/$u/main; touch bemb/$u/main.DONE; ln -sfn nc bemb/$u/delay; done
cp -n $B0/bemb/ath_hufeat_main.csv bemb/
for k in cg_main_bones.npz cg_main_ribs3.npz v15_main r15_main; do ln -sfn $B0/frac/$k frac/$k; done
ln -sfn ../nc15/v15/emb15 frac/v15_delay; ln -sfn ../nc15/r15/emb15 frac/r15_delay
cp -n $B0/osteo_v15/data/cg_main_ov.npz $B0/osteo_v15/data/l1hu_levels_main.csv $B0/osteo_v15/data/merlin_ov.npz osteo_v15/data/
ln -sfn $B0/gs6/main gs6/main
echo SETUP_OK
