#!/bin/bash
# gs15 self-running pipeline on A6000 (膽結石 1.5 mm GB crop unit; train/val only; test never read).
# Usage (in ~/Desktop/Tony/VLM/OWL/mw_exp): setsid nohup bash outputs_final/gallstones15/scripts/run_gs15.sh NW >> outputs_final/gallstones15/logs/driver.log 2>&1 < /dev/null &
#  NW crop workers (CPU nice 10, 2 threads; CT ssh-cat from CT-rate HDD read-only; 3 mm masks + GS6 masks local) -> crops/
#  (+ Mac crop workers reading the Passport, pushed into crops/ by gs15_mac_sync.sh)
#  1 GPU extractor (frozen TS Dataset297 encoder, 6 windows, polls crops/) -> emb/
#  tier0 embedded -> interim eval (probe -> feats -> [cnn] -> stack -> report: out_tier0/, stack_tier0/, report.md)
#  everything embedded -> full eval (out/, stack/, report.md) -> FINAL_DONE
# Stop: touch outputs_final/gallstones15/STOP (crop workers stop after the current case; extractor/driver stop; Mac sync propagates STOP).
# Resume: rm STOP STOP_EXTRACT; rm ct/*; delete claims/<id> without crops/<id>.npz AND without emb/<id>.npz; rerun this script.
MW=$HOME/Desktop/Tony/VLM/OWL/mw_exp; R=$MW/outputs_final/gallstones15; S=$R/scripts; PY=$HOME/miniconda3/envs/radar/bin/python; NW=${1:-6}
TS=$HOME/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
G6=$MW/outputs_final/region_units/gallstones/gs6_seg
cd $R; mkdir -p logs crops emb claims ct
echo "DRIVER_START $(date) pid $$ NW $NW"
if [ "$NW" -gt 0 ]; then for i in $(seq 0 $((NW-1))); do
  setsid nohup bash -c "while [ ! -f $R/STOP ] && [ ! -f $R/STOP_a6 ] && [ ! -f $R/CROP_DONE_a6_w$i ]; do nice -n 10 $PY $S/gs15_crop.py --ids $R/inputs/ids_a6.txt --maskmap $R/inputs/maskmap.tsv --gs6 $G6 --root $R --ct-ssh --tag a6 --wid $i --max-cases 300 --threads 2; sleep 2; done" > logs/crop_a6_w$i.log 2>&1 < /dev/null &
  echo "crop worker $i loop pid $!"; sleep 1
done; fi
setsid nohup bash -c "cd $MW; while [ ! -f $R/STOP ] && [ ! -f $R/EXTRACT_DONE ]; do MW_ROOT=. $PY $S/gs15_extract.py --crops $R/crops --out $R/emb --ts-model-dir $TS --windows $S/gs15_windows.yaml --poll --done-flag $R/CROP_ALL_DONE --device cuda --vram-gb 12 --threads 2 >> $R/logs/extract.log 2>&1 && [ -f $R/CROP_ALL_DONE ] && touch $R/EXTRACT_DONE; sleep 10; done" > logs/extract_loop.log 2>&1 < /dev/null &
echo "extract loop pid $!"
run_eval () {  # $1 = tier0 | full
  if [ "$1" = tier0 ]; then OUT=out_tier0; TAG=stack_tier0; SEL=0; else OUT=out; TAG=stack; SEL=all; fi
  $PY $S/gs15_status.py $R labels $R/inputs/labels_$1.csv $SEL
  MW_SRC=$S NT=4 nice -n 5 $PY $S/gs15_probe.py gs15 $R/emb $MW/outputs_final/region_units/probe_inputs/main_gs.npz $R/inputs/labels_$1.csv $R/$OUT --device cuda --vram-gb 12 --variants U,U_org,U_gb3 > logs/probe_$1.log 2>&1
  nice -n 5 $PY $S/gs15_feats.py $R > logs/feats_$1.log 2>&1
  if [ -f $S/gs15_cnn.py ] && [ ! -f $R/STOP_CNN ]; then nice -n 5 $PY $S/gs15_cnn.py $R $R/inputs/labels_$1.csv $R/cnn_$1 > logs/cnn_$1.log 2>&1; fi
  nice -n 5 $PY $S/gs15_stack.py $MW/outputs_final $R $1 > logs/$TAG.log 2>&1
  echo "EVAL_$1_DONE $(date)"; tail -25 logs/$TAG.log
}
while [ ! -f $R/STOP ]; do
  $PY $S/gs15_status.py $R status
  if [ ! -f $R/CROP_ALL_DONE ] && $PY $S/gs15_status.py $R all_cropped; then touch $R/CROP_ALL_DONE; echo "CROP_ALL_DONE $(date)"; fi
  if [ ! -f $R/TIER0_EVAL_DONE ] && $PY $S/gs15_status.py $R tier0_done; then
    echo "TIER0 complete $(date) -> interim eval"; run_eval tier0 && touch $R/TIER0_EVAL_DONE
  fi
  if [ -f $R/EXTRACT_DONE ]; then
    echo "ALL extracted $(date) -> full eval"; run_eval full && touch $R/FINAL_DONE; echo "FINAL_DONE $(date)"; break
  fi
  sleep 300
done
echo "DRIVER_EXIT $(date)"
