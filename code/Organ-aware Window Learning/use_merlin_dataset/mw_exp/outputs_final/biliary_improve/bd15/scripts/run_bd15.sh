#!/bin/bash
# bd15 self-running pipeline, ADAPTED FOR A6000 (11-06 10:40; Spark original = run_bd15.sh.spark_orig).
# Usage (in ~/Desktop/Tony/VLM/OWL/mw_exp): setsid nohup bash outputs_final/biliary_improve/bd15/scripts/run_bd15.sh NW > outputs_final/biliary_improve/bd15/logs/driver.log 2>&1 < /dev/null &
# A6000: 3 mm masks read locally (inputs/maskmap.tsv -> outputs/masks, outputs_vert15/masks3_extra); original CT still ssh-cat from CT-rate HDD (read-only).
#  NW crop workers (CPU, nice 10, 2 threads, CT + 3 mm mask pulled from CT-rate HDD by ssh, read-only) -> crops/
#  1 GPU extractor (frozen TS Dataset297 encoder, 5 windows, polls crops/) -> emb/
#  tier0 finished -> interim probe (out_tier0) + stack (stack_tier0) + report; everything finished -> full probe (out) + stack + report -> FINAL_DONE
# Stop: touch outputs_final/biliary_improve/bd15/STOP (crop workers stop after the current case; extractor + driver stop when they see STOP / STOP_EXTRACT).
# Resume: rm STOP STOP_EXTRACT; rm ct/*; delete claims/<id> without crops/<id>.npz; rerun this script (done work is skipped).
MW=$HOME/Desktop/Tony/VLM/OWL/mw_exp; R=$MW/outputs_final/biliary_improve/bd15; S=$R/scripts; PY=$HOME/miniconda3/envs/radar/bin/python; NW=${1:-6}
TS=$HOME/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
cd $R; mkdir -p logs crops emb claims ct
echo "DRIVER_START $(date) pid $$ NW $NW"
for i in $(seq 0 $((NW-1))); do
  setsid nohup bash -c "while [ ! -f $R/STOP ] && [ ! -f $R/CROP_DONE_w$i ]; do nice -n 10 $PY $S/bd15_crop.py --ids $R/inputs/ids_prio.csv --maskmap $R/inputs/maskmap.tsv --root $R --wid $i --max-cases 300 --threads 2; sleep 2; done" > logs/crop_w$i.log 2>&1 < /dev/null &
  echo "crop worker $i loop pid $!"; sleep 1
done
setsid nohup bash -c "cd $MW; while [ ! -f $R/STOP ] && [ ! -f $R/EXTRACT_DONE ]; do MW_ROOT=. $PY $S/bd15_extract.py --crops $R/crops --out $R/emb --ts-model-dir $TS --windows windows.yaml --poll --done-flag $R/CROP_ALL_DONE --device cuda --vram-gb 24 --threads 4 >> $R/logs/extract.log 2>&1 && [ -f $R/CROP_ALL_DONE ] && touch $R/EXTRACT_DONE; sleep 10; done" > logs/extract_loop.log 2>&1 < /dev/null &
echo "extract loop pid $!"
run_eval () {  # $1 = tier0 | full
  if [ "$1" = tier0 ]; then OUT=out_tier0; TAG=stack_tier0; SEL=0; else OUT=out; TAG=stack; SEL=all; fi
  $PY $S/bd15_status.py $R labels $R/inputs/labels_$1.csv $SEL
  MW_SRC=$S NT=6 nice -n 5 $PY $S/bd15_probe.py bd15 $R/emb $R/inputs/main_bd.npz $R/inputs/labels_$1.csv $R/$OUT --device cuda --vram-gb 16 > logs/probe_$1.log 2>&1
  nice -n 5 $PY $S/bd15_stack.py $R $R/inputs/labels_$1.csv $TAG > logs/$TAG.log 2>&1
  $PY $S/bd15_report.py $R stack_tier0 stack > logs/report.log 2>&1
  echo "EVAL_$1_DONE $(date)"; tail -12 logs/$TAG.log
}
while [ ! -f $R/STOP ]; do
  $PY $S/bd15_status.py $R status
  if [ ! -f $R/CROP_ALL_DONE ]; then
    nd=0; for i in $(seq 0 $((NW-1))); do [ -f $R/CROP_DONE_w$i ] && nd=$((nd+1)); done
    [ $nd -eq $NW ] && touch $R/CROP_ALL_DONE && echo "CROP_ALL_DONE $(date)"
  fi
  if [ ! -f $R/TIER0_EVAL_DONE ] && $PY $S/bd15_status.py $R tier0_done; then
    echo "TIER0 complete $(date) -> interim eval"; run_eval tier0 && touch $R/TIER0_EVAL_DONE
  fi
  if [ -f $R/EXTRACT_DONE ]; then
    echo "ALL extracted $(date) -> full eval"; run_eval full && touch $R/FINAL_DONE; echo "FINAL_DONE $(date)"; break
  fi
  sleep 300
done
echo "DRIVER_EXIT $(date)"
