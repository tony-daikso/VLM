#!/bin/bash
# GS6 step 4-5 (CPU, nice 19) on a host that has the 3 mm prep+mask of the cases:
#  gs6_fix.py (full-res GB -> 3 mm prep/mask)  ->  gs_hu.py, ag_feat.py, gs1_feat.py on the fixed dirs  ->  gs5_crop.py
# usage: gs6_post.sh ids.txt prep_dirs(comma) mask_dirs(comma) seg_dir work_dir scripts_dir   (PY env var = python)
set -e
IDS=$1; PD=$2; MD=$3; SEG=$4; W=$5; SC=$6; PY=${PY:-~/miniconda3/envs/radar/bin/python}; NP=${NP:-3}
mkdir -p "$W"
nice -n 19 $PY "$SC/gs6_fix.py" "$IDS" "$PD" "$MD" "$SEG" "$W"
ls "$W/prep" | sed 's/.npz//' > "$W/fixed_ids.txt"
nice -n 19 $PY "$SC/gs_hu.py" "$W/fixed_ids.txt" "$W/prep" "$W/hu_fix.csv" $NP
nice -n 19 $PY "$SC/ag_feat.py" "$W/fixed_ids.txt" "$W/prep" "$W/masks" "$W/ag_fix.csv" $NP x 0.3355,0.7002,0.6623
nice -n 19 $PY "$SC/gs1_feat.py" "$W/fixed_ids.txt" "$W/prep" "$W/g1_fix.csv" $NP
nice -n 19 $PY "$SC/gs5_crop.py" "$W/fixed_ids.txt" "$W/crops" "$W/prep" "$W/masks" $NP
echo GS6_POST_DONE
