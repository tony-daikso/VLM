#!/bin/bash
# spine15_ft: crop CTs relayed into buf/ (from the Mac Passport, relay_passport.sh) with 4 procs; delete each CT after.
# stop: touch $D/STOP_BUF
D=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw; cd $D; PY=~/miniconda3/envs/radar/bin/python
while [ ! -f STOP_BUF ]; do
  ls buf | grep -E '^AC.*\.nii\.gz$' | sed 's/.nii.gz//' > .buf_ids.txt
  if [ -s .buf_ids.txt ]; then
    RAW15=$D/buf nice -n 10 $PY scripts/spine15_crop.py .buf_ids.txt $D 4 $D/../vert15_seg15 ~/OWL/mw_exp/outputs/masks $D/masks3 >> logs/crop_buf.log 2>&1
    while read c; do rm -f buf/$c.nii.gz; done < .buf_ids.txt
  else sleep 10; fi
done
