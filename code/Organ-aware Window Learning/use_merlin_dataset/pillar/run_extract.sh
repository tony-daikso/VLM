#!/usr/bin/env bash
# Watchdog wrapper for extract.py (train -> val only; test is NOT passed / gated by --allow-test).
# Launch:  setsid nohup bash run_extract.sh > /media/daikso/Tony/pillar/merlin/watchdog.log 2>&1 < /dev/null &
# Restarts extract.py (resumable) when it exits non-zero or when the heartbeat is stale > STALE seconds.
# Each run lives in its own process group so the GPU process + its CPU workers are killed together.
set -u
CODE=/media/daikso/Tony/pillar/code
OUT=/media/daikso/Tony/pillar/merlin
PY=/media/daikso/Tony/envs/pillar/bin/python
HB=$OUT/heartbeat
STALE=${STALE:-1200}
WORKERS=${WORKERS:-6}
THREADS=${THREADS:-2}
MAX_RESTARTS=${MAX_RESTARTS:-30}
mkdir -p "$OUT"
echo $$ > "$OUT/watchdog.pid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HOME=/media/daikso/Tony/pillar/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
restarts=0
while :; do
  date +%s > "$HB"
  echo "[watchdog $(date '+%F %T')] start extract (restart $restarts)"
  setsid "$PY" "$CODE/extract.py" --out "$OUT" --workers "$WORKERS" --threads "$THREADS" \
      --heartbeat "$HB" >> "$OUT/extract.log" 2>&1 < /dev/null &
  pid=$!
  echo $pid > "$OUT/extract.pid"
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    hb=$(cut -d' ' -f1 "$HB" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - ${hb:-0} ))
    if [ "$age" -gt "$STALE" ]; then
      echo "[watchdog $(date '+%F %T')] heartbeat stale ${age}s -> kill pgid $pid"
      kill -TERM -- -$pid 2>/dev/null; sleep 20; kill -KILL -- -$pid 2>/dev/null
      break
    fi
  done
  wait $pid; rc=$?
  # make sure no orphan workers keep RAM/VRAM
  kill -KILL -- -$pid 2>/dev/null
  echo "[watchdog $(date '+%F %T')] extract exited rc=$rc"
  if [ "$rc" -eq 0 ] && tail -n 3 "$OUT/extract.log" | grep -q ALL_DONE; then
    echo "[watchdog] ALL_DONE"; break
  fi
  if [ "$rc" -eq 5 ]; then echo "[watchdog] systemic GPU failure (rc=5), not restarting"; break; fi
  restarts=$((restarts+1))
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then echo "[watchdog] too many restarts, giving up"; break; fi
  sleep 30
done
