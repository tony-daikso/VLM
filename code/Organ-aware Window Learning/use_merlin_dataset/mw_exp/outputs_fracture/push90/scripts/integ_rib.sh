#!/bin/bash
# push90 rib (+PE) integration variants; 3 in parallel. usage: integ_rib.sh PEDIR
F=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture; P=$F/push90; S=$P/scripts/integ_run.sh; PF="--add P15FT=$F/pelvis15_ft/final@pel"; PE="--add PE=$1@vert"
$S r_a6repl --add RVRS=$P/cand/mixA6@rib $PF & $S r_a6mix3 --add RVRS=$P/cand/mix3A6@rib $PF & $S r_a6add --add RVRS=$F/rib_straight/cand/mix15@rib --add A6=$P/cand/a6_side@rib $PF & wait
$S r_a6only --add RVRS=$P/cand/a6_side@rib $PF & $S rp_a6repl --add RVRS=$P/cand/mixA6@rib $PF $PE & $S rp_a6mix3 --add RVRS=$P/cand/mix3A6@rib $PF $PE & wait
$S rp_a6add --add RVRS=$F/rib_straight/cand/mix15@rib --add A6=$P/cand/a6_side@rib $PF $PE & $S rp_a6only --add RVRS=$P/cand/a6_side@rib $PF $PE & wait
echo "$(date "+%m-%d %H:%M") INTEG_RIB_DONE" >> $P/logs/push90.log
