#!/bin/bash
# =====================================================================================================
# integrated_best: rebuild the TEST inputs of every integrated-best model with the SAME code as train/val.
# !!! DO NOT RUN until the user has locked finding_organ_map.yaml / the configuration (test is held). !!!
# Nothing in this script was executed on test. It refuses to run unless IB_TEST_LOCKED=yes-locked-by-user.
#
# usage: IB_TEST_LOCKED=yes-locked-by-user bash rebuild_test_features.sh <test_ids.txt> <outputs_root> <work_dir> [nproc]
#   <outputs_root> has prep/{id}.npz and masks/{id}.nii.gz (3 mm, TS --fast)  e.g. ~/Desktop/Tony/VLM/OWL/mw_exp/outputs
#   run once per host that holds prep/masks (A6000, CT-rate) and concatenate the csvs (same as for train/val).
# Steps: 1 image features (CPU) -> 2 probe heads (refit on all train, same recipe/seeds) -> 3 apply saved stacks.
# =====================================================================================================
set -euo pipefail
[ "${IB_TEST_LOCKED:-}" = "yes-locked-by-user" ] || { echo "refusing: test split is held until the user locks the config"; exit 1; }
IDS=$1; O=$2; W=$3; NP=${4:-4}
IB=$(cd "$(dirname "$0")/.." && pwd)                 # .../outputs_final/integrated_best
C=$IB/scripts/analysis_code                           # frozen copies of the feature / OOF scripts
P=${PY:-~/miniconda3/envs/radar/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $W/feat $W/heads $W/scores
N="nice -n 19 $P"

# ---------- step 1: image features (identical scripts/arguments as for train/val) ----------
$N $C/hydronephrosis/hn_hu.py        $IDS $O                    $W/feat/hydronephrosis_hu.csv $NP
$N $C/atherosclerosis/ath_hu.py      $IDS $O/prep $O/masks      $W/feat/atherosclerosis_hu.csv $NP
$N $C/free_air/fa_air.py             $IDS $O/prep $O/masks      $W/feat/free_air_air.csv $NP
$N $C/ascites/asc_hu.py              $IDS $O                    $W/feat/ascites_hu.csv $NP
$N $C/aaa/aaa_geom.py                $IDS $O/prep $O/masks      $W/feat/aaa_geom.csv $NP
$N $C/absent_gb/ag_feat.py           $IDS $O/prep $O/masks      $W/feat/absent_gb_feat.csv $NP
$N $C/bowel_obstruction/sbo_feat.py  $IDS $O/prep $O/masks      $W/feat/sbo_feat.csv $NP
$N $C/splenomegaly/sp_geom.py        $IDS $O/prep $O/masks      $W/feat/splenomegaly_geom.csv $NP
$N $C/osteopenia/op_hu.py            $IDS $O/prep $O/masks      $W/feat/osteopenia_corehu.csv $NP
$N $C/steatosis/hs_hu.py             $IDS $O/prep $O/masks      $W/feat/steatosis_hu.csv $NP
$N $C/appendicitis/ap_hu.py          $IDS $O/prep               $W/feat/appendicitis_hu.csv $NP
# osteopenia bundle = per-vertebra geometry/HU of src/fracture_features.py (same code as train/val):
$N $IB/../../src/fracture_features.py --outputs $O --csv $W/feat/fracture_features_test.csv
#   then build the bundle columns exactly like analysis_code/osteopenia/op_bundle.py (TRV -> test ids; metadata
#   columns Age/Gender/contrast/phase/manufacturer/kvp/slicethickness are dropped by ib_apply, never model inputs).
# gallstones (GS5 G5c) and biliary (BD5) features: their own pipelines (same code), see report.md section 5:
$N $C/gallstones/gs_hu.py            $IDS $O/prep               $W/feat/gs_hu.csv $NP
$N $C/gallstones/gs1_feat.py         $IDS $O/prep               $W/feat/gs1_feat3mm.csv $NP
$N $C/biliary/bd_hu.py               $IDS $O/prep $O/masks      $W/feat/bd_hu.csv $NP
$N $C/biliary/bd3_feat.py            $IDS $O/prep $O/masks      $W/feat/bd3_tube.csv $NP
#   GPU parts (not CPU features): gs5_crop.py/gs5_extract.py/gs5_probe.py (gbx head), gs3b_oof.py (g3b head),
#   bd5_crop.py/bd5_extract (porta embedding) + bd5_probe.py; fracture: outputs_vert15 final_eval (Exp J S3).

# ---------- step 2: probe heads (same OOF script; refit on all train; scores the 'test' rows) ----------
# Needs the embedding matrix INCLUDING test rows (built by the main pipeline after unlock). Run each analysis dump
# script with its *_notest.npz path replaced by that matrix and its split filter widened to train+test, then e.g.:
#   $P $IB/scripts/ib_test_heads.py $C/hydronephrosis/hn_oof.py $IB/../../src <dir with kidney_emb + labels incl. test rows>
# Heads used by the saved stacks (see models/<finding>_features.json):
#   hydronephrosis E3_late+E3_late_max | atherosclerosis E3_late | free_air E3_late | ascites see json |
#   abdominal_aortic_aneurysm E3_late | surgically_absent_gallbladder E3_late | bowel_obstruction see json |
#   splenomegaly E3_late | osteopenia see json | hepatic_steatosis E3_late+E3_late_concat_diff |
#   appendicitis E3_late+E3_late_colon_sb
# Write one csv per finding to $W/heads/<finding>.csv (study_id + head columns, names as in the json).

# ---------- step 3: apply the saved stacks ----------
ap() { $P $IB/scripts/ib_apply.py $IB $1 $W/heads/$1.csv $2 $W/scores/$1.csv; }
ap hydronephrosis                $W/feat/hydronephrosis_hu.csv
ap atherosclerosis               $W/feat/atherosclerosis_hu.csv
ap free_air                      $W/feat/free_air_air.csv
ap ascites                       $W/feat/ascites_hu.csv
ap abdominal_aortic_aneurysm     $W/feat/aaa_geom.csv
ap surgically_absent_gallbladder $W/feat/absent_gb_feat.csv
ap bowel_obstruction             $W/feat/sbo_feat.csv
ap splenomegaly                  $W/feat/splenomegaly_geom.csv
ap osteopenia                    $W/feat/osteopenia_bundle.csv,$W/feat/osteopenia_corehu.csv
ap hepatic_steatosis             $W/feat/steatosis_hu.csv
ap appendicitis                  $W/feat/appendicitis_hu.csv
echo "gallstones: gallstones_improve/gs5/stack/stack_G5c.joblib via gs_stack2.py feature assembly;"
echo "biliary: models/biliary_* (see report.md); fracture: outputs_vert15/final_eval S3 (final_stack.py config.json)."
