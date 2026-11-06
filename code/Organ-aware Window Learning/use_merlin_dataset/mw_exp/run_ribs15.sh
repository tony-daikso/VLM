#!/bin/bash
# Experiment L workers with recycling. usage: ./run_ribs15.sh <python> <ids.txt> <out> <wid_from> <wid_to> <max_cases> <ct_local|-> <mask_dir>...
cd "$(dirname "$0")"
PY=$1; IDS=$2; OUT=$3; W0=$4; W1=$5; MAXC=$6; CTL=$7; shift 7; MASKS="$*"
mkdir -p "$OUT/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CTARG=""; [ "$CTL" != "-" ] && CTARG="--ct-local $CTL"
for i in $(seq $W0 $W1); do
  setsid nohup bash -c "while [ ! -f '$OUT/STOP' ] && [ ! -f '$OUT/DONE_w$i' ]; do nice -n 5 '$PY' src/rib15_full.py extract --ids '$IDS' --out '$OUT' --boxes outputs_ribs15/rib_boxes.npz --mask-dirs $MASKS $CTARG --wid $i --max-cases $MAXC --threads 2; sleep 2; done" \
    >> "$OUT/logs/w$i.log" 2>&1 < /dev/null &
  echo "w$i loop pid $!"
done
