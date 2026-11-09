#!/bin/bash
# Run on the Mac: copy the small inputs (agent B targets, agent A OOF scores / measurements / calibration, metadata)
# to A6000 and build method 4 train/val jsonl THERE (the jsonl hold doctor sentences: they stay on A6000, not the Mac).
# Train + val only; build_jsonl.py refuses test rows.   usage: bash build_data.sh
set -euo pipefail
MD=${MERLIN_DATA:-/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin}
RG=$MD/report_gen
M4=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4
HERE=$(cd "$(dirname "$0")" && pwd)
for f in "$RG/targets/targets_train.jsonl" "$RG/targets/targets_val.jsonl" "$RG/oof/out/scores_trval_prob.csv" \
         "$RG/oof/out/measure_trval.csv" "$RG/method1/measure_calibration.json" "$MD/disease14/metadata.csv"; do
  [ -s "$f" ] || { echo "missing $f"; exit 1; }
done
ssh A6000 "mkdir -p $M4/inputs $M4/data $M4/code"
rsync -a "$RG/targets/targets_train.jsonl" "$RG/targets/targets_val.jsonl" "$RG/oof/out/scores_trval_prob.csv" \
      "$RG/oof/out/measure_trval.csv" "$RG/method1/measure_calibration.json" "$MD/disease14/metadata.csv" A6000:$M4/inputs/
rsync -a "$HERE"/*.py "$HERE"/*.sh A6000:$M4/code/
ssh A6000 "cd $M4/code && ~/miniconda3/envs/radar/bin/python build_jsonl.py \
  --targets $M4/inputs/targets_train.jsonl $M4/inputs/targets_val.jsonl \
  --scores $M4/inputs/scores_trval_prob.csv --measure $M4/inputs/measure_trval.csv \
  --calib $M4/inputs/measure_calibration.json --meta $M4/inputs/metadata.csv \
  --manifest $M4/features/manifest.csv --feat-root $M4/features/ --out $M4/data && wc -l $M4/data/*.jsonl"
