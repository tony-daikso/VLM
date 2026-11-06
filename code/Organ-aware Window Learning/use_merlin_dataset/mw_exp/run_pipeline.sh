#!/bin/bash
# Runs steps after masks: patches -> extract -> qc -> probe -> evaluate. Waits for a running masks.py first.
set -e
cd "$(dirname "$0")"
PY=${PY:-~/miniconda3/envs/radar/bin/python}
while pgrep -f "src/masks.py" > /dev/null; do sleep 30; done
$PY src/masks.py            # picks up anything left
$PY src/patches.py
$PY src/extract.py
$PY src/qc.py
$PY src/probe.py "$@"
$PY src/evaluate.py --min-test-pos ${MIN_TEST_POS:-10}
echo PIPELINE_DONE
