#!/usr/bin/env bash
# Option A driver (A6000, CPU only). Waits until the CT-rate Pillar extraction is finished (ALL_DONE in extract.log, or the
# watchdog is gone), packs the compact features on CT-rate (nice 19, 1 thread), pulls them, then runs the full analysis.
# Launch: setsid nohup bash run_full.sh > ~/Desktop/Tony/VLM/OWL/pillar_optA/out/run_full.log 2>&1 < /dev/null &
set -u
W=~/Desktop/Tony/VLM/OWL/pillar_optA
PY=~/miniconda3/envs/radar/bin/python
CT=daikso@192.168.0.189
PM=/media/daikso/Tony/pillar/merlin
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date '+%F %T')] waiting for extraction"
while :; do
  st=$(ssh -o BatchMode=yes $CT "tail -n 3 $PM/extract.log | grep -c ALL_DONE; kill -0 \$(cat $PM/watchdog.pid) 2>/dev/null && echo alive || echo dead; ls $PM/feats | grep -vc tmp" 2>/dev/null | tr '\n' ' ')
  echo "[$(date '+%F %T')] $st"
  set -- $st
  if [ "${1:-0}" -ge 1 ] || [ "${2:-alive}" = "dead" ]; then break; fi
  sleep 300
done
echo "[$(date '+%F %T')] extraction finished -> pack"
ssh -o BatchMode=yes $CT "cd /media/daikso/Tony/pillar/code/optionA && OMP_NUM_THREADS=1 nice -n 19 /media/daikso/Tony/envs/pillar/bin/python pack_feats.py --feats $PM/feats --out $PM/packed" || exit 1
rsync -a $CT:$PM/packed/ $W/packed/ || exit 1
ssh -o BatchMode=yes $CT "wc -l < $PM/failures.jsonl 2>/dev/null || echo 0" | sed 's/^/failures.jsonl lines: /'
cd $W/code
echo "[$(date '+%F %T')] cls14"; nice -n 10 $PY optA.py cls14 --workers 6 || exit 1
echo "[$(date '+%F %T')] kidney"; OMP_NUM_THREADS=6 nice -n 10 $PY optA.py kidney --workers 6 || exit 1
echo "[$(date '+%F %T')] vert"; OMP_NUM_THREADS=6 nice -n 10 $PY optA.py vert --workers 6 || exit 1
echo "[$(date '+%F %T')] cheap"; OMP_NUM_THREADS=4 nice -n 10 $PY optA.py cheap || exit 1
$PY report.py > /dev/null && echo "[$(date '+%F %T')] OPTA_ALL_DONE"
