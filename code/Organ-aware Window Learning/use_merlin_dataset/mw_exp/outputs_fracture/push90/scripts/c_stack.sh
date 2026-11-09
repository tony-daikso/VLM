#!/bin/bash
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; MIX=$F/rib_straight/cand/mix15; PF=$F/pelvis15_ft/final; S=$F/push90/scripts/integ_run.sh
B="--add RVRS=$MIX@rib --add P15FT=$PF@pel"
$S base_repro $B
$S c_fn0 $B --wcsv $F/push90/C/w_fn0.csv
$S c_fn0fp03 $B --wcsv $F/push90/C/w_fn0_fp03.csv
$S c_all03 $B --wcsv $F/push90/C/w_all03.csv
