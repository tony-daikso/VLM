#!/bin/bash
# CG10k gallstone features (gs_hu.py, gs1_feat.py; reproduced bit-exact on 8 Merlin cases), after the biliary features.
P=~/miniconda3/envs/radar/bin/python; C=~/OWL/cg10k_best/scripts/analysis_code/gallstones
until [ -f /media/daikso/CTDATA/cg10k/owl_feat/delay/bd3_feat.done ]; do sleep 60; done
for S in main delay; do D=~/OWL/cg10k_best/data/$S; W=/media/daikso/CTDATA/cg10k/owl_feat/$S
  OMP_NUM_THREADS=1 nice -n 19 $P $C/gs_hu.py $W/ids_ok.txt $D/prep $W/gs_hu.csv 8
  OMP_NUM_THREADS=1 nice -n 19 $P $C/gs1_feat.py $W/ids_ok.txt $D/prep $W/gs1_feat.csv 8 && date > $W/gsfeat.done
done
