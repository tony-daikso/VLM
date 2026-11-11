#!/bin/bash
# Option B (arm P_gt): build the full Pillar store once the CT-rate extraction is finished, move it to A6000, verify.
# Runs on A6000 (it can ssh to CT-rate), low priority, no GPU:
#   wait for ALL_DONE in extract.log (or the watchdog gone) -> pack_pillar.py on CT-rate (nice 19, ionice -c3, 1 thread)
#   into the CT-rate staging dir -> npz spot check (50 rows) -> rsync (nice / ionice) to A6000 $DST -> md5 of every file
#   on both sides -> READY marker -> delete the CT-rate staging copy and the small A6000 preflight store.
# One copy only on A6000: $DST (~66 GB: grid16 64 GB + units 1.6 GB + small arrays).
# Launch: setsid nohup bash run_store.sh > ~/Desktop/Tony/VLM/OWL/report_gen/m4/code_p/logs/run_store.log 2>&1 < /dev/null &
set -u
TONY=/home/advantech/Desktop/Tony
CT=daikso@192.168.0.189
PM=/media/daikso/Tony/pillar/merlin
STAGE=$PM/p_arm_store
CODE=/media/daikso/Tony/pillar/code/p_arm
CPY=/media/daikso/Tony/envs/pillar/bin/python
DST=$TONY/VLM/OWL/pillar_feats
PRE=$TONY/VLM/OWL/pillar_feats_preflight
ts() { date '+%F %T'; }
echo "[$(ts)] waiting for extraction"
while :; do
  st=$(ssh -o BatchMode=yes $CT "tail -n 3 $PM/extract.log | grep -c ALL_DONE; kill -0 \$(cat $PM/watchdog.pid) 2>/dev/null && echo alive || echo dead; ls $PM/feats | grep -vc tmp" 2>/dev/null | tr '\n' ' ')
  echo "[$(ts)] $st"
  set -- $st
  if [ "${1:-0}" -ge 1 ] || [ "${2:-alive}" = "dead" ]; then break; fi
  sleep 600
done
echo "[$(ts)] extraction finished -> pack on CT-rate"
ssh -o BatchMode=yes $CT "cd $CODE && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 $CPY pack_pillar.py --feats $PM/feats --out $STAGE && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 $CPY test_p_arm.py npz --store $STAGE --feats $PM/feats | head -6" || { echo "pack failed"; exit 1; }
ssh -o BatchMode=yes $CT "python3 -c \"import json; m=json.load(open('$STAGE/pack_meta.json')); print('n', m['n'], 'done', m['n_done'], 'bad', len(m['bad']), 'no mask', len(m['mask_missing']))\""
avail=$(df -BG --output=avail $TONY | tail -1 | tr -dc 0-9)
need=$(ssh -o BatchMode=yes $CT "du -sBG $STAGE | cut -f1 | tr -dc 0-9")
echo "[$(ts)] A6000 free ${avail} GB, store ${need} GB"
[ "$avail" -gt $((need + 60)) ] || { echo "not enough disk on A6000 (keep >= 60 GB spare)"; exit 1; }
mkdir -p $DST
echo "[$(ts)] rsync -> $DST"
for i in 1 2 3; do
  nice -n 19 ionice -c3 rsync -a --partial --inplace $CT:$STAGE/ $DST/ && break
  echo "rsync try $i failed"; sleep 60
done
echo "[$(ts)] md5 check"
ssh -o BatchMode=yes $CT "cd $STAGE && nice -n 19 ionice -c3 md5sum *.npy ids.txt pack_meta.json" | sort -k2 > $DST/.md5_src
(cd $DST && nice -n 19 ionice -c3 md5sum *.npy ids.txt pack_meta.json | sort -k2 > $DST/.md5_dst)
if cmp -s $DST/.md5_src $DST/.md5_dst; then
  date '+%F %T' > $DST/READY
  echo "[$(ts)] verified -> READY; deleting CT-rate staging $STAGE and A6000 preflight store $PRE"
  ssh -o BatchMode=yes $CT "rm -rf $STAGE $PM/p_arm_preflight"
  rm -rf $PRE
  du -sh $DST
  echo "[$(ts)] STORE_DONE"
else
  echo "md5 mismatch:"; diff $DST/.md5_src $DST/.md5_dst; exit 1
fi
