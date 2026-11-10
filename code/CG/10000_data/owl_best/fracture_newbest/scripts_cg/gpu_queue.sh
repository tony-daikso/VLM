#!/bin/bash
# CT-rate RTX 3080 queue for fracture a6x10w2_tv7 CG10k (main C+, 6,970): one GPU job at a time, each waits for its CPU inputs
# (logs/chain.log markers from cpu_chain.sh). Steps skip themselves when their done-file exists (resume = rerun). stop: touch $W/STOP_GPU
W=/media/daikso/Tony/VLM/data/CT/CG/10000_data/owl_frac_v3; M=/media/daikso/Tony/VLM/data/CT/merlin; PY=~/miniconda3/envs/radar/bin/python
SRC=~/OWL/mw_exp/src; L=$W/logs; cd $W; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
log(){ echo "$(date '+%m-%d %H:%M') $*" >> $L/gpu_queue.log; }
waitfor(){ until grep -q "$1" $L/chain.log 2>/dev/null; do [ -f STOP_GPU ] && exit 0; sleep 60; done; }
step(){ local done=$1; shift; [ -f $done ] && return 0; [ -f STOP_GPU ] && exit 0; log "start $*"; "$@"; local rc=$?; log "end rc=$rc $1"; return $rc; }

# 1) FT_sp full-train refit (Merlin 3 mm spine packs; fp32 recipe, bs 8 = 4 x 2 accumulation) -> ftsp/model_full_s{0,1}.pt
ftsp_train(){ nice -n 5 $PY scripts/frac_ft_full.py --mode full --branch spine --labels ftsp/labels.csv --packs $M/frac15_crops/pack_a6.npz $M/frac15_crops/pack_ct.npz \
  --out ftsp --src $SRC --enc 297 --train-from 2 --pool lse --epochs 8 --seeds 2 --seed0 0 --bs 8 --lr 1e-4 --accum 4 --vram-gb 9.5 --pred-bs 2 --threads 3 > $L/ftsp_train.log 2>&1 && touch ftsp/TRAIN_DONE; }
step ftsp/TRAIN_DONE ftsp_train

# 2) PE features (pe_feat.py, Dataset292 st0-2, bf16 as Merlin)
waitfor "pe15 done"; cp ids_main.txt pe15/ids.txt
pefeat(){ nice -n 5 $PY $M/push90/pe15/scripts/pe_feat.py pe15/crops pe15/feats --ids ids_main.txt --src $SRC --amp bf16 --vram-gb 7 > $L/pe_feat.log 2>&1 && touch pe15/FEAT_DONE; }
step pe15/FEAT_DONE pefeat

# 3) FT_sp15 CG score with the saved 5 fold models (frac15_ft.py --score in a sandbox: CG rows appended as 'val')
waitfor "sp15 done"
ft15(){ # name run_dir script crops_merlin crops_cg labels extra...
  local n=$1 run=$2 scr=$3 cm=$4 cc=$5 lb=$6; shift 6; local S=$W/ft15/$n; mkdir -p $S/out $S/crops
  cp -n $run/fold*.npz $S/out/; for f in $run/model_f*.pt; do ln -sf $f $S/out/; done
  [ -f $S/crops/.linked ] || { find $cm -maxdepth 1 -name "*.npz" ! -name "*tmp*" -exec ln -sf {} $S/crops/ \; ; find $cc -maxdepth 1 -name "*.npz" ! -name "*tmp*" -exec ln -sf {} $S/crops/ \; ; touch $S/crops/.linked; }
  $PY - $lb $W/ids_main.txt $S/labels_cg.csv <<'PYX'
import sys, pandas as pd
lab = pd.read_csv(sys.argv[1]); assert set(lab.split) <= {"train", "val"}
cg = [l.strip() for l in open(sys.argv[2]) if l.strip()]
add = pd.DataFrame({c: 0 for c in lab.columns if c != "study_id"}, index=range(len(cg))); add.insert(0, "study_id", cg)
add["split"] = "val"; add["fold"] = -1
for c in ("in_sitewise", "tierA"):
    if c in add: add[c] = False
pd.concat([lab, add[lab.columns]]).to_csv(sys.argv[3], index=False)
PYX
  nice -n 5 $PY $scr --labels $S/labels_cg.csv "$@" --out $S/out --src $SRC --score > $L/ft15_$n.log 2>&1 && touch $S/SCORE_DONE
}
step ft15/sp15/SCORE_DONE ft15 sp15 $M/frac15_raw/runs/s0_tf2lse $M/frac15_raw/scripts/frac15_ft.py $M/frac15_raw/crops $W/sp15/crops $M/frac15_raw/labels.csv \
  --crops $W/ft15/sp15/crops --train-from 2 --pool lse --mask m --wins base,bone,soft --amp none --seeds 1 --seed0 0 --vram-gb 9 --pred-bs 2 --threads 3 --workers 3

# 4) P15FT CG score, both seeds (pv15_ft.py --score, bf16 as the Merlin scoring)
waitfor "pv15 done"
TS297=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
for s in 0 1; do
  step ft15/pv15_s$s/SCORE_DONE ft15 pv15_s$s $M/pelvis15_ft/runs/full_297s_s$s $M/pelvis15_ft/scripts/pv15_ft.py $M/pelvis15/roi $W/pv15/roi $M/pelvis15/labels.csv \
    --roi $W/ft15/pv15_s$s/crops --ts $TS297 --train-from 2 --seeds 1 --seed0 0 --pool-ids $M/pelvis15_ft/pool_ids.txt --side-csv $M/pelvis15/pel_side.csv \
    --cell 2 --minvox 64 --maxshape 176,144,240 --fixed --shift 6 --amp bf16 --target t_pel --wins base,bone --vram-gb 9 --threads 3 --workers 4
done

# 5) FT_sp CG score with the full-train models
waitfor "sp3 done"
ftsp_cg(){ nice -n 5 $PY scripts/frac_ft_full.py --mode score --branch spine --labels none --packs sp3/pack_cg.npz --out ftsp --src $SRC --enc 297 \
  --train-from 2 --pool lse --vram-gb 9.5 --pred-bs 2 --threads 3 --score-out ftsp/cg_scores.csv > $L/ftsp_cg.log 2>&1 && touch ftsp/CG_DONE; }
step ftsp/CG_DONE ftsp_cg

# 6) rib feats3 (rib_feat3.py --maxst 1 --bone --nsec 24; always bf16 autocast)
waitfor "rib_prep done"
feat3(){ nice -n 5 $PY $M/rib_ft/scripts/rib_feat3.py rib/roi rib/feats3 --maxst 1 --bone --nsec 24 --once --src $SRC > $L/feat3.log 2>&1 && touch rib/FEAT3_DONE; }
step rib/FEAT3_DONE feat3

# 7) straightened-rib features: rs15 (rs_feat.py, fp32) and a6/w2 (rs_feat2.py stages 0-2 base+bone, fp32)
waitfor "rs_tile done"
rsf(){ nice -n 5 $PY scripts/rs_feat.py rs/tiles rs/feats --ids ids_main.txt --src $SRC --vram-gb 9 > $L/rs_feat.log 2>&1 && touch rs/FEAT_DONE; }
step rs/FEAT_DONE rsf
rsf2(){ nice -n 5 $PY $M/push90/scripts/rs_feat2.py rs/tiles rs/feats_v2 --ids ids_main.txt --stages 0,1,2 --wins base,bone --src $SRC --vram-gb 9 > $L/rs_feat2.log 2>&1 && touch rs/FEAT2_DONE; }
step rs/FEAT2_DONE rsf2
log GPU_QUEUE_DONE
