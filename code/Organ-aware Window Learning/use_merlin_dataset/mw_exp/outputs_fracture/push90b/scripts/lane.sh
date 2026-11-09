#!/bin/bash
# push90b generic GPU lane (resumable): runs specs sequentially, 2 tries each; a spec failing twice stops the lane.
# usage: lane.sh HOSTTAG BASE PY SCRIPT "name|outdir|labels|feats|args" ...   (paths relative to BASE)  stop: touch BASE/STOP
# done marker = <outdir>/summary.json.  logs: BASE/logs/<name>.log, events: BASE/logs/push90b.log
H=$1; B=$2; PY=$3; SC=$4; shift 4; cd "$B" || exit 1; mkdir -p logs
for spec in "$@"; do
  IFS='|' read -r n od lab fe ar <<< "$spec"; ok=0
  for try in 1 2; do
    [ -f "$od/summary.json" ] && { ok=1; break; }; [ -f STOP ] && exit 0
    echo "$(date "+%m-%d %H:%M") $H $n start try $try" >> logs/push90b.log
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 5 $PY $SC $lab $fe $od $ar >> logs/$n.log 2>&1
    rc=$?; echo "$(date "+%m-%d %H:%M") $H $n rc=$rc $(grep PROBE logs/$n.log | tail -1 | cut -c1-260)" >> logs/push90b.log
    [ -f "$od/summary.json" ] && { ok=1; break; }
    [ -f STOP ] && exit 0; sleep 30
  done
  [ $ok = 1 ] || { echo "$(date "+%m-%d %H:%M") $H $n FAILED twice -> lane stops" >> logs/push90b.log; exit 1; }
done
echo "$(date "+%m-%d %H:%M") $H lane done: $(for s in "$@"; do echo -n "${s%%|*} "; done)" >> logs/push90b.log
