#!/bin/bash
# BD5 full train+val porta-hepatis crop-unit embedding on A6000 (GPU) + probe + stack.
# RUN ONLY AFTER Exp J has finished on A6000: no vert15_full.py processes left (or outputs_vert15/DONE_w0 and DONE_w1 exist).
# Run from ~/Desktop/Tony/VLM/OWL/mw_exp :   bash outputs_final/biliary_improve/scripts/run_bd5.sh
set -e
cd ~/Desktop/Tony/VLM/OWL/mw_exp
B=outputs_final/biliary_improve
if pgrep -f "src/vert15_full.py" > /dev/null; then echo "Exp J still running on A6000 -> abort"; exit 1; fi
if [ -f outputs_final/A6000_GPU_LOCK ]; then echo "A6000_GPU_LOCK held (gallstones GS5/GS6) -> abort: $(cat outputs_final/A6000_GPU_LOCK)"; exit 1; fi
# the gallstones chain (gs_gpu_chain.sh) waits up to 45 min after J for bd5_extract.py to appear, then waits for it.
PY=~/miniconda3/envs/radar/bin/python
# 1) crops (made on CT-rate SSD by bd5_crop.py; ~7 GB) -> A6000
mkdir -p $B/crops
rsync -a daikso@192.168.0.189:OWL/mw_exp/analysis_biliary_tmp/crops/ $B/crops/
# 2) GPU extraction (frozen TS encoder, 5 windows, stages 1-3, mean|max per region; ~0.2-0.3 s/case on A6000)
MW_ROOT=. nice -n 10 $PY $B/scripts/bd5_extract.py --crops $B/crops --out $B/emb_porta \
   --ids outputs_final/analysis_biliary/ids_trval.txt --device cuda --config config_final.yaml > $B/bd5_extract.log 2>&1
# 3) probe (train 5-fold OOF + full-train -> val), same recipe/folds as bd_oof.py
cp src/probe.py src/common.py $B/scripts/ 2>/dev/null || true
NT=4 nice -n 19 $PY $B/scripts/bd5_probe.py \
   "outputs_final/results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz" \
   $B/emb_porta outputs_final/analysis_biliary/labels_trval.csv outputs_final/analysis_biliary/oof_train.csv $B/out > $B/bd5_probe.log 2>&1
# 4) re-stack with the porta embedding scores (bd_stack.py picks up bd5_oof_train.csv / bd5_val.csv in WORK)
cp $B/out/bd5_oof_train.csv $B/out/bd5_val.csv $B/work/
nice -n 19 $PY $B/scripts/bd_stack.py $B/work $B/stack_bd5 > $B/stack_bd5.log 2>&1
echo BD5_ALL_DONE
