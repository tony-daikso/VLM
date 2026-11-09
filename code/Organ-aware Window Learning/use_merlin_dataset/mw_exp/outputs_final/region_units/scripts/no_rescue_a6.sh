#!/bin/bash
# Exp O rescue on A6000 (4090 GPU died ~06:58 with CUDA_ERROR_LAUNCH_FAILED / "Unknown Error").
# Finishes the last unit bowel_obstruction_r2 (gastric outlet/duodenum, code go) and the failed frpp probe, all on A6000.
# stop: touch outputs_final/region_units/STOP_RESCUE_O (checked between steps; extraction is resumable, existing emb skipped)
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units
PY=/home/advantech/miniconda3/envs/radar/bin/python
TS=$HOME/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
U=bowel_obstruction_r2; MW=~/Desktop/Tony/VLM/OWL/mw_exp
st(){ [ -f STOP_RESCUE_O ] && { echo "STOPPED at $1 $(date)"; exit 0; }; }
mkdir -p $U/logs $U/crops_ct fracture_pelvis_r2/logs
echo "START $(date)"
# 1) CT-rate crops (read-only pull; CT-rate CPU/disk only, its GPU untouched)
rsync -a --bwlimit=60000 --exclude '*.tmp*' daikso@192.168.0.189:OWL/mw_exp/region_units_tmp/$U/crops/ $U/crops_ct/ && echo "PULL_DONE $(ls $U/crops_ct | grep -c npz) $(date)"
st extract
# 2) GPU extraction (existing emb skipped), A6000 crops then CT-rate crops
for d in crops crops_ct; do
  (cd $MW && MW_ROOT=. OMP_NUM_THREADS=3 nice -n 19 $PY outputs_final/region_units/scripts/no_extract.py --crops outputs_final/region_units/$U/$d \
     --out outputs_final/region_units/$U/emb --ts-model-dir $TS --windows windows.yaml --threads 3 --vram-gb 8 --device cuda) >> $U/logs/extract_a6.log 2>&1
done
echo "EXTRACT_DONE emb=$(ls $U/emb | grep -c npz) ERR=$(grep -c ^ERR $U/logs/extract_a6.log) $(date)"
st probe
# 3) probes on A6000 GPU
pr(){ MW_SRC=$MW/src NT=3 OMP_NUM_THREADS=3 nice -n 19 $PY scripts/no_probe.py $1 $2/emb probe_inputs/main_$1.npz probe_inputs/labels_$1.csv $2/$3 --device cuda --vram-gb 8 > $2/logs/probe_$1.log 2>&1; tail -4 $2/logs/probe_$1.log; }
pr frpp fracture_pelvis_r2 out_pel
st stack
OMP_NUM_THREADS=4 nice -n 19 $PY scripts/no_stack.py fracture_pelvis_r2 > fracture_pelvis_r2/stack.log 2>&1; grep -v -i warn fracture_pelvis_r2/stack.log | tail -16
echo "POST_DONE fracture_pelvis_r2 $(date)"
pr go $U out
OMP_NUM_THREADS=4 nice -n 19 $PY scripts/no_stack.py $U > $U/stack.log 2>&1; grep -v -i warn $U/stack.log | tail -16
echo "POST_DONE $U $(date)"
echo "ALL_DONE $(date)"
