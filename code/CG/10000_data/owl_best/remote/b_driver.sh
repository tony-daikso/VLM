#!/bin/bash
# Tier B driver on A6000 (main set): when a unit's CG embeddings are done -> refit unit-probe heads (Merlin train only,
# GPU, VRAM cap 8 GB) -> refit/save the selected stack and score CG (bstack_apply.py).  Appendicitis RLQ via cg_brlq.py.
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python; RS=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
w(){ until [ -f bemb/$1/main.DONE ]; do sleep 60; done; }
probe(){ c=$1; s=$2; [ -f bprobe/$c/out/val_scores.csv ] && [ -f bprobe/$c/PROBE_OK ] && return
  rm -rf bprobe/$c/out; $P cg_bprobe.py prep $c bemb/$c/main && $P cg_bprobe.py run $c $s --device cuda --vram-gb 8 > bprobe_$c.log 2>&1 && touch bprobe/$c/PROBE_OK; }
w hn;  probe hn $RS/scripts/ru_probe.py
$P bstack_apply.py hydronephrosis hydronephrosis "REF＋全部單元頭" hn best/main/hydronephrosis_heads.csv feat/main/hydronephrosis_hu.csv > bstack_hydronephrosis.log 2>&1
w fa;  probe fa $RS/scripts/ru_probe.py
$P bstack_apply.py free_air free_air "REF＋全部單元頭" fa best/main/free_air_heads.csv feat/main/free_air_air.csv > bstack_free_air.log 2>&1
w ath; probe ath $RS/scripts/nu_probe.py
$P bstack_apply.py atherosclerosis atherosclerosis "REF＋全部單元頭＋HU" ath best/main/atherosclerosis_heads.csv feat/main/atherosclerosis_hu.csv $RS/atherosclerosis/hu_feats.csv bemb/ath_hufeat_main.csv > bstack_atherosclerosis.log 2>&1
w rlq; $P cg_brlq.py bemb/rlq/main > brlq.log 2>&1
w asc; probe asc $RS/scripts/ru_probe.py
until [ -f best/main/ascites_heads.csv ]; do sleep 120; done
$P bstack_apply.py ascites ascites "REF＋U_org" asc best/main/ascites_heads.csv feat/main/ascites_hu.csv > bstack_ascites.log 2>&1
echo B_DRIVER_DONE
