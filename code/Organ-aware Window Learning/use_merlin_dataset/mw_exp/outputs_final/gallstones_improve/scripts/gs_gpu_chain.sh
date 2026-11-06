#!/bin/bash
# GS5 + GS6 GPU chain on A6000 (run from outputs_final/gallstones_improve, nice 19).
#  0. pull the CT-rate GS5 crops (CPU, small files)
#  1. wait until Exp J has finished on A6000 (no vert15_full.py process left)
#  2. biliary coordination: never two GPU extractions at once -> wait while bd5_extract.py runs; after J, give the biliary
#     porta-hepatis extraction up to WAIT_BD5 s to appear (it goes first), then wait for it to finish
#  3. GS5: gs5_extract.py (GB crop units, 5 windows, mean+max) on cuda
#  4. GS6: gs6_seg.py (TS full-res Dataset291) on the streamed 1.5 mm crops (--consume) until DONE_SENDING
# Lock: ../A6000_GPU_LOCK (holder + PID) while 3/4 run. Stop: touch STOP_CHAIN (between steps) / kill this PID + child.
cd "$(dirname "$0")/.." || exit 1
PY=~/miniconda3/envs/radar/bin/python
MW=~/Desktop/Tony/VLM/OWL/mw_exp
WAIT_BD5=${WAIT_BD5:-2700}
log() { echo "$(date '+%F %T') $*"; }
stop() { [ -f STOP_CHAIN ] && { log "STOP_CHAIN"; rm -f ../A6000_GPU_LOCK; exit 0; }; }
log "chain start pid $$"
rsync -a daikso@192.168.0.189:OWL/gs5/crops/ gs5/crops/ && log "pulled CT-rate crops: $(ls gs5/crops | wc -l)"
while pgrep -f "src/vert15_full.py" > /dev/null; do stop; sleep 300; done
log "J finished on A6000"
rsync -a daikso@192.168.0.189:OWL/gs5/crops/ gs5/crops/ && log "crops now $(ls gs5/crops | wc -l)"
t0=$(date +%s); seen=0
while true; do
  stop
  if pgrep -f "bd5_extract.py" > /dev/null; then seen=1; sleep 120; continue; fi
  [ "$seen" = 1 ] && { log "biliary bd5_extract finished"; break; }
  [ $(( $(date +%s) - t0 )) -gt "$WAIT_BD5" ] && { log "no biliary extraction within ${WAIT_BD5}s -> go"; break; }
  sleep 60
done
stop
echo "gallstones GS5/GS6 chain pid $$ since $(date '+%F %T')" > ../A6000_GPU_LOCK
ls gs5/crops | sed 's/.npz//' > gs5/crop_ids.txt
log "GS5 extract start ($(wc -l < gs5/crop_ids.txt) crops)"
(cd "$MW" && nice -n 19 $PY outputs_final/gallstones_improve/scripts/gs5_extract.py --crops outputs_final/gallstones_improve/gs5/crops \
   --out outputs_final/gallstones_improve/gs5/emb --ids outputs_final/gallstones_improve/gs5/crop_ids.txt --device cuda) > gs5/extract.log 2>&1
log "GS5 extract exit $? : $(tail -1 gs5/extract.log)"
stop
log "GS6 seg start"
nice -n 19 $PY scripts/gs6_seg.py --inbox gs6/inbox --out gs6/seg --device cuda --consume > gs6/seg.log 2>&1
log "GS6 seg exit $? : $(tail -1 gs6/seg.log)"
rm -f ../A6000_GPU_LOCK
log "CHAIN_DONE"
