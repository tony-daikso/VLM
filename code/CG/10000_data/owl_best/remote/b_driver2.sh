#!/bin/bash
# Biliary BD5 and gallstones N_U drivers on A6000 (main set), each step waits for its inputs.
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python; CT=daikso@192.168.0.189; F=/media/daikso/CTDATA/cg10k/owl_feat
RS=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
biliary(){
  until [ -f bemb/bd5/main.DONE ] && ssh $CT "test -f $F/main/bd3_feat.done"; do sleep 120; done
  for s in main delay; do ssh $CT "test -f $F/$s/bd3_feat.done" && scp -q $CT:$F/$s/bd3_feat.csv feat/$s/; done
  $P cg_biliary.py bd5 bemb/bd5/main > bbd5.log 2>&1
  $P cg_biliary.py stack main > bbd_stack_main.log 2>&1
  echo BILIARY_DONE >> b_driver2.log
}
gall(){
  until ! pgrep -f "patched_gs_oof.py" >/dev/null && [ -f heads/R/analysis_gallstones/scores_val_descriptive.csv ]; do sleep 60; done
  until ssh $CT "test -f $F/main/gsfeat.done"; do sleep 120; done
  for s in main delay; do ssh $CT "test -f $F/$s/gsfeat.done" && scp -q $CT:$F/$s/gs_hu.csv $CT:$F/$s/gs1_feat.csv feat/$s/; done
  ssh $CT "test -f $F/delay/gsfeat.done" || { until ssh $CT "test -f $F/delay/gsfeat.done"; do sleep 60; done; scp -q $CT:$F/delay/gs_hu.csv $CT:$F/delay/gs1_feat.csv feat/delay/; }
  $P cg_gs.py g3b > gs/g3b_driver.log 2>&1
  $P cg_gs.py gbx > gs/gbx_driver.log 2>&1
  until [ -f bemb/gs/main.DONE ]; do sleep 120; done
  rm -rf bprobe/gs/out; $P cg_bprobe.py prep gs bemb/gs/main && $P cg_bprobe.py run gs $RS/scripts/nu_probe.py --device cuda --vram-gb 8 > bprobe_gs.log 2>&1
  $P cg_gs.py stack main > gs/stack_main.log 2>&1
  echo GALL_DONE >> b_driver2.log
}
biliary & gall & wait
