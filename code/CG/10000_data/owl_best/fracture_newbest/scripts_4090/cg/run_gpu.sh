#!/bin/bash
# CG feature extraction on the rented 4090s (CG crops only; Merlin images never here). usage: run_gpu.sh GPU
# claim-based: tasks = rs_feat2 x10 chunks, rs_feat x10 chunks, rib_feat3 x4 chunks; claim = mkdir claims/<task>; done = claims/<task>/ok
# GPU1 joins after its Merlin queue (gpu1 QUEUE_DONE). outputs out/{feats_v2,feats,feats3}; stop: touch ~/Tony/cg/STOP
cd ~/Tony/cg; . ~/Tony/venv/bin/activate; g=$1; mkdir -p logs out
export CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log(){ echo "$(date "+%m-%d %H:%M") gpu$g $*" >> logs/run.log; }
[ $g = 1 ] && until grep -q "gpu1 QUEUE_DONE" ~/Tony/logs/queue.log; do sleep 30; done
until [ -f TILES_DONE ]; do sleep 30; done
run(){ local t=$1; shift; [ -f STOP ] && exit 0; mkdir claims/$t 2>/dev/null || return 0; log "start $t"; "$@" > logs/$t.log 2>&1; local rc=$?; log "end $t rc=$rc"; [ $rc = 0 ] && touch claims/$t/ok; }
for c in $(ls chunks | grep tiles_); do run f2_$c python scripts/rs_feat2.py tiles out/feats_v2 --ids chunks/$c --stages 0,1,2 --wins base,bone --src src --vram-gb 20; done
for c in $(ls chunks | grep tiles_); do run f1_$c python scripts/rs_feat.py tiles out/feats --ids chunks/$c --src src --vram-gb 20; done
until [ -f ROI_DONE ]; do sleep 30; done
for c in $(ls chunks | grep f3_); do run f3_$c python scripts/rib_feat3.py roi out/feats3 --ids chunks/$c --maxst 1 --bone --nsec 24 --once --src src; done
log DONE
