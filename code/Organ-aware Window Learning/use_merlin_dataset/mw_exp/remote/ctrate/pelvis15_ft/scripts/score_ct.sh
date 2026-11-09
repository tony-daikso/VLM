#!/bin/bash
# CT-rate 3080: when all 5 fold files + fold models of a seed dir are present, fill val + non-pool train rows with --score (bf16)
# and assemble runs/full_297s_s<seed>/{oof_train,val_scores,summary}. agent pelvis15_ft 11-08. stop: touch runs/STOP
cd /media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft; P=../pelvis15; PY=~/miniconda3/envs/radar/bin/python
TS=$(ls -d ~/.totalsegmentator/nnunet/results/Dataset297_*/nnUNet* | head -1)
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/full.log; }
until grep -q "ct lane done\|ct job .* FAILED" logs/full.log; do [ -f runs/STOP ] && exit 0; sleep 120; done
for s in 0 1; do
  d=runs/full_297s_s$s
  until [ "$(ls $d/fold?.npz 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls $d/model_f?_s$s.pt 2>/dev/null | wc -l)" -ge 5 ]; do [ -f runs/STOP ] && exit 0; sleep 120; done
  [ -f $d/SCORE_DONE ] && continue
  for try in 1 2; do
    log "score $d try $try"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY scripts/pv15_ft.py --labels $P/labels.csv --roi $P/roi --out $d \
      --ts $TS --src ~/OWL/mw_exp/src --pool-ids pool_ids.txt --side-csv $P/pel_side.csv --fixed --maxshape 176,144,240 --score \
      --amp bf16 --vram-gb 9 --workers 4 --threads 3 >> logs/score_s$s.log 2>&1
    rc=$?; log "score $d rc=$rc $(grep 'PV15FT train OOF' logs/score_s$s.log | tail -1)"
    [ $rc = 0 ] && { touch $d/SCORE_DONE; break; }
  done
  [ -f $d/SCORE_DONE ] || { log "score $d FAILED twice"; exit 1; }
done
log SCORE_ALL_DONE
