#!/bin/bash
# gs15 A6000 eval driver (CPU only; replaces the eval part of run_gs15.sh after the 12:40 instruction "no new GPU work on
# A6000"). Crop loops + the GPU extract loop started by run_gs15.sh keep running independently.
# tier0 embedded -> labels_tier0.csv + REQ_tier0 -> (Mac gs15_remote.sh runs the probe + CNN on the 4090 and drops
# out_tier0/PROBE_DONE here; fallback after 180 min: CPU probe here) -> feats -> stack (CPU) -> report.md -> TIER0_EVAL_DONE.
# All embedded (EXTRACT_DONE) -> same with full -> FINAL_DONE.  Stop: touch STOP (stops crops/extract too) or STOP_EVAL (only this).
MW=$HOME/Desktop/Tony/VLM/OWL/mw_exp; R=$MW/outputs_final/gallstones15; S=$R/scripts; PY=$HOME/miniconda3/envs/radar/bin/python
cd $R; echo "EVAL_DRIVER_START $(date) pid $$"
run_eval () {
  if [ "$1" = tier0 ]; then OUT=out_tier0; TAG=stack_tier0; SEL=0; else OUT=out; TAG=stack; SEL=all; fi
  $PY $S/gs15_status.py $R labels $R/inputs/labels_$1.csv $SEL && touch $R/REQ_$1
  for i in $(seq 1 180); do [ -f $R/$OUT/PROBE_DONE ] && break; [ -f $R/STOP_EVAL ] && return 1; sleep 60; done
  if [ ! -f $R/$OUT/PROBE_DONE ]; then
    echo "no 4090 probe after 180 min -> CPU probe $(date)"
    MW_SRC=$S NT=6 nice -n 5 $PY $S/gs15_probe.py gs15 $R/emb $MW/outputs_final/region_units/probe_inputs/main_gs.npz $R/inputs/labels_$1.csv $R/$OUT --device cpu --variants U,U_org,U_gb3 > logs/probe_$1_cpu.log 2>&1 && touch $R/$OUT/PROBE_DONE
  fi
  nice -n 5 $PY $S/gs15_feats.py $R > logs/feats_$1.log 2>&1
  nice -n 5 $PY $S/gs15_stack.py $MW/outputs_final $R $1 > logs/$TAG.log 2>&1
  echo "EVAL_$1_DONE $(date)"; grep -E "SELECTED|^head|REF_rb corr" logs/$TAG.log
}
while [ ! -f $R/STOP ] && [ ! -f $R/STOP_EVAL ]; do
  $PY $S/gs15_status.py $R status
  if [ ! -f $R/CROP_ALL_DONE ] && $PY $S/gs15_status.py $R all_cropped; then touch $R/CROP_ALL_DONE; echo "CROP_ALL_DONE $(date)"; fi
  if [ ! -f $R/TIER0_EVAL_DONE ] && $PY $S/gs15_status.py $R tier0_done; then
    echo "TIER0 complete $(date) -> interim eval"; run_eval tier0 && touch $R/TIER0_EVAL_DONE
  fi
  if [ -f $R/EXTRACT_DONE ]; then echo "ALL extracted $(date) -> full eval"; run_eval full && touch $R/FINAL_DONE; echo "FINAL_DONE $(date)"; break; fi
  sleep 300
done
echo "EVAL_DRIVER_EXIT $(date)"
