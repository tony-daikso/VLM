#!/bin/bash
# CG10k: integrated_best image features (same frozen analysis_code; reproduced bit-exact on 8 Merlin cases per script, 2026-11-06)
# usage: run_feat.sh <set: main|delay> [nproc]
set -u
S=$1; NP=${2:-6}; P=~/miniconda3/envs/radar/bin/python; C=~/OWL/cg10k_best/scripts/analysis_code
O=$HOME/OWL/cg10k_best/data/$S; W=/media/daikso/CTDATA/cg10k/owl_feat/$S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ls $O/prep | sed "s/.npz//" | sort > $W/ids.txt
comm -12 $W/ids.txt <(ls $O/masks | sed "s/.nii.gz//" | sort) > $W/ids_ok.txt
I=$W/ids_ok.txt; echo "$(date) $S cases $(wc -l < $I)"
N="nice -n 19 $P"
r(){ f=$1; shift; [ -s $W/$f.done ] && { echo "skip $f"; return; }; echo "$(date +%T) start $f"; "$@" && date > $W/$f.done; echo "$(date +%T) end $f"; }
r hydronephrosis_hu  $N $C/hydronephrosis/hn_hu.py       $I $O                $W/hydronephrosis_hu.csv $NP
r atherosclerosis_hu $N $C/atherosclerosis/ath_hu.py     $I $O/prep $O/masks  $W/atherosclerosis_hu.csv $NP
r free_air_air       $N $C/free_air/fa_air.py            $I $O/prep $O/masks  $W/free_air_air.csv $NP
r ascites_hu         $N $C/ascites/asc_hu.py             $I $O                $W/ascites_hu.csv $NP
r aaa_geom           $N $C/aaa/aaa_geom.py               $I $O/prep $O/masks  $W/aaa_geom.csv $NP
r absent_gb_feat     $N $C/absent_gb/ag_feat.py          $I $O/prep $O/masks  $W/absent_gb_feat.csv $NP x 0.336,0.700,0.661
r sbo_feat           $N $C/bowel_obstruction/sbo_feat.py $I $O/prep $O/masks  $W/sbo_feat.csv $NP
r splenomegaly_geom  $N $C/splenomegaly/sp_geom.py       $I $O/prep $O/masks  $W/splenomegaly_geom.csv $NP
r steatosis_hu       $N $C/steatosis/hs_hu.py            $I $O/prep $O/masks  $W/steatosis_hu.csv $NP
r appendicitis_hu    $N $C/appendicitis/ap_hu.py         $I $O/prep           $W/appendicitis_hu.csv $NP
r osteopenia_corehu  $N $C/osteopenia/op_hu.py           $I $O/prep $O/masks  $W/osteopenia_corehu.csv $NP
echo "$(date) ALL DONE $S"; date > $W/ALL_DONE
