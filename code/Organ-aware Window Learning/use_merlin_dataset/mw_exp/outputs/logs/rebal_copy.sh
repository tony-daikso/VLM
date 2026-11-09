#!/bin/bash
# copy A6000 inputs for emb_bones_ids_from_a6000.txt to CT-rate outputs_from_a6000/ (prep -> masks -> patches)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs
for d in prep masks patches; do
  case $d in prep) ext=npz;; masks) ext=nii.gz;; patches) ext=json;; esac
  sed "s/\$/.$ext/" emb_bones_ids_from_a6000.txt > /tmp/rebal_$d.lst
  date; rsync -a --ignore-existing --files-from=/tmp/rebal_$d.lst $d/ daikso@192.168.0.189:OWL/mw_exp/outputs_from_a6000/$d/ && echo "done $d"
done
date
