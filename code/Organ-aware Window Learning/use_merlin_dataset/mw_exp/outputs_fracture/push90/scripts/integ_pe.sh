#!/bin/bash
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; P=$F/push90; S=$P/scripts/integ_run.sh; B="--add RVRS=$F/rib_straight/cand/mix15@rib --add P15FT=$F/pelvis15_ft/final@pel"
$S pe_ctxh $B --add PE=$P/cand/pe_ctx_head@vert & $S pe_ctxs $B --add PE=$P/cand/pe_ctx_side@vert & wait
$S pe_plain $B --add PE=$P/cand/pe_plain@vert & $S pe_mix $B --add PE=$P/cand/pe_mix@vert & wait
