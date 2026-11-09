#!/bin/bash
# after the c8 full feature pass: extract c6 (9 mm cells) for all cases (ensemble member). stop: touch feats_c6/STOP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_straight
until grep -qE "FEAT_DONE|abort" logs/feat_all.log; do sleep 60; done
grep -q FEAT_DONE logs/feat_all.log || exit 1
nice -n 5 ~/miniconda3/envs/radar/bin/python scripts/rs_feat.py tiles feats_c6 --cell 6 --ns 40 --vram-gb 5 > logs/feat_c6_all.log 2>&1
