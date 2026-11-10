#!/bin/bash
# Tier B, delay set: after a6_bpipe.sh has embedded the delay crops, re-run the unit-probe refits with main+delay CG rows
# (same Merlin train fit -> identical main scores) and apply the saved-recipe stacks to the delay set (CG_PREFIX=D_).
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python; RS=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
until grep -q A6_BPIPE_DONE a6_bpipe.log; do sleep 120; done
for c in hn fa asc ath; do
  for f in $(ls bprobe/$c/emb | grep "^D_"); do rm -f bprobe/$c/emb/$f; done
  rm -rf bprobe/$c/out; s=ru_probe.py; [ $c = ath ] && s=nu_probe.py
  $P cg_bprobe.py prep $c bemb/$c/main bemb/$c/delay && $P cg_bprobe.py run $c $RS/scripts/$s --device cuda --vram-gb 8 > bprobe_${c}_md.log 2>&1
done
export CG_PREFIX=D_
$P bstack_apply.py hydronephrosis hydronephrosis "REF＋全部單元頭" hn best/delay/hydronephrosis_heads.csv feat/delay/hydronephrosis_hu.csv > bstack_hydronephrosis_d.log 2>&1
$P bstack_apply.py free_air free_air "REF＋全部單元頭" fa best/delay/free_air_heads.csv feat/delay/free_air_air.csv > bstack_free_air_d.log 2>&1
$P bstack_apply.py ascites ascites "REF＋U_org" asc best/delay/ascites_heads.csv feat/delay/ascites_hu.csv > bstack_ascites_d.log 2>&1
$P bstack_apply.py atherosclerosis atherosclerosis "REF＋全部單元頭＋HU" ath best/delay/atherosclerosis_heads.csv feat/delay/atherosclerosis_hu.csv $RS/atherosclerosis/hu_feats.csv bemb/ath_hufeat_delay.csv > bstack_atherosclerosis_d.log 2>&1
rm -rf bprobe/rlq/out; $P cg_brlq.py bemb/rlq/main bemb/rlq/delay > brlq_md.log 2>&1
echo B_DELAY_DONE
