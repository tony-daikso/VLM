#!/bin/bash
# push90 B: crop CTs relayed into buf/ from the Mac Passport (Mac relay_pe.sh) with 4 procs; delete each CT after. stop: touch /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/STOP_BUF
cd /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15; PY=~/miniconda3/envs/radar/bin/python
while [ ! -f STOP_BUF ]; do
  ls buf | grep -E '^AC.*\.nii\.gz$' | sed 's/.nii.gz//' > .buf_ids.txt
  if [ -s .buf_ids.txt ]; then
    RAW15=/media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/buf OMP_NUM_THREADS=1 nice -n 10 $PY scripts/pe15_crop.py .buf_ids.txt /media/daikso/Tony/VLM/data/CT/merlin/push90/pe15 4 /media/daikso/Tony/VLM/data/CT/merlin/vert15_seg15 ~/OWL/mw_exp/outputs/masks /media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/masks3 >> logs/crop_buf.log 2>&1
    while read c; do rm -f buf/$c.nii.gz; done < .buf_ids.txt
  else sleep 10; fi
done
