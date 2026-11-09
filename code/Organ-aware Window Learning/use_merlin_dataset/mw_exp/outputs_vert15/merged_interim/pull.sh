#!/bin/bash
# Exp J interim: READ-ONLY pull of finished emb15/seg15 .npz from A6000 / CT-rate / Spark (no *.tmp*), first copy wins.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_vert15
CT=daikso@192.168.0.189; SP=daikso_spark@192.168.0.95
R="nice -n 19 rsync -a --ignore-existing --exclude=*.tmp* --include=*.npz --exclude=* --bwlimit=20000"
SSH="ssh -o BatchMode=yes"
date +%T
for k in emb15 seg15; do
  $R emb15/../$k/ merged_interim/$k/
  $R -e "$SSH" $CT:/home/daikso/OWL/expJ_ct/out_vert15/$k/ merged_interim/$k/
  $R -e "$SSH" $SP:/home/daikso_spark/OWL/expJ/out_vert15/$k/ merged_interim/$k/
  echo $k $(ls merged_interim/$k | wc -l) $(date +%T)
done
echo PULL_DONE
