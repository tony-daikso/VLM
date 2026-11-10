#!/bin/bash
# after the claim runner: GPU0 -> score CG with saved probe runs (pe, pe2, rs15, a6) then feats3 poller part 0/2;
# GPU1 -> feats3 poller part 1/2. pollers stop when out/feats3/LAST exists (touched once ROI_XFER_DONE) and nothing is left.
cd ~/Tony/cg; . ~/Tony/venv/bin/activate; g=$1; export CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log(){ echo "$(date "+%m-%d %H:%M") gpu$g $*" >> logs/run.log; }
until grep -q "gpu$g DONE" logs/run.log; do sleep 20; done
if [ $g = 0 ]; then
  until [ -f ~/Tony/runs/a6/f_w2/summary_full.json ]; do sleep 20; done
  log "score_cg start"; ~/Tony/score_cg.sh 0 pe pe2 rs15 a6; log "score_cg end"
fi
mkdir -p out/feats3; ( until [ -f ROI_XFER_DONE ]; do sleep 30; done; touch out/feats3/LAST ) &
log "feat3 poll part $g/2 start"; python scripts/rib_feat3.py roi out/feats3 --ids /dev/null --part $g/2 --maxst 1 --bone --nsec 24 --src src > logs/feat3_poll_$g.log 2>&1; log "feat3 poll end rc=$?"
