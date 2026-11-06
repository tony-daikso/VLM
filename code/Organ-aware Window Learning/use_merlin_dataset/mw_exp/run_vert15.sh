#!/bin/bash
# Experiment J workers with recycling. usage: ./run_vert15.sh <python> <ids.txt> <out> <n_workers> <max_cases> <ct_local|-> <mask_dir>...
cd "$(dirname "$0")"
PY=$1; IDS=$2; OUT=$3; N=$4; MAXC=$5; CTL=$6; shift 6; MASKS="$*"
mkdir -p "$OUT/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CTARG=""; [ "$CTL" != "-" ] && CTARG="--ct-local $CTL"
for i in $(seq 0 $((N-1))); do
  setsid nohup bash -c "while [ ! -f '$OUT/STOP' ] && [ ! -f '$OUT/DONE_w$i' ]; do '$PY' src/vert15_full.py extract --ids '$IDS' --out '$OUT' --mask-dirs $MASKS $CTARG --wid $i --max-cases $MAXC --threads 2; sleep 2; done" \
    > "$OUT/logs/w$i.log" 2>&1 < /dev/null &
  echo "w$i loop pid $!"
done
