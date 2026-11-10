#!/bin/bash
# CG10k biliary BD1/BD3 features (bd3_feat.py = bd_hu features + tube + porta/HDL; reproduced bit-exact on 8 Merlin cases)
P=~/miniconda3/envs/radar/bin/python; C=~/OWL/cg10k_best/scripts/analysis_code/biliary
for S in main delay; do D=~/OWL/cg10k_best/data/$S; W=/media/daikso/CTDATA/cg10k/owl_feat/$S
  OMP_NUM_THREADS=1 nice -n 19 $P $C/bd3_feat.py $W/ids_ok.txt $D/prep $D/masks $W/bd3_feat.csv 8 && date > $W/bd3_feat.done
done
