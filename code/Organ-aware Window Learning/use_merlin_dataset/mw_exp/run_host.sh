#!/bin/bash
# Start N stage-A workers (+ one extract --watch) on this host for the streamed full run.
# usage: ./run_host.sh <config.yaml> <python> <inbox_dir> <n_workers> [run_extract=1] [max_cases_per_worker_life=30]
# Each worker exits after max_cases cases and is restarted by its loop (returns fragmented CUDA cache memory).
cd "$(dirname "$0")"
CFG=$1; PY=$2; INBOX=$3; N=$4; EXT=${5:-1}; MAXC=${6:-30}
OUT=$(grep '^output_dir:' "$CFG" | awk '{print $2}')
mkdir -p "$OUT/logs" "$INBOX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 0 $((N-1))); do
  setsid nohup bash -c "while [ ! -f '$INBOX/STOP' ]; do '$PY' src/worker.py --config '$CFG' --inbox '$INBOX' --wid $i --max-cases $MAXC; done" \
    > "$OUT/logs/worker_$i.log" 2>&1 < /dev/null &
done
if [ "$EXT" = 1 ]; then
  setsid nohup "$PY" src/extract.py --config "$CFG" --watch > "$OUT/logs/extract.log" 2>&1 < /dev/null &
fi
echo "started $N workers (extract=$EXT, recycle every $MAXC cases) -> $OUT"
