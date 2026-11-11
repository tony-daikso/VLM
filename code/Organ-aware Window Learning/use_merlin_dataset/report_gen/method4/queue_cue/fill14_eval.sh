#!/bin/bash
# Improvement 1 ("fill all 14", method4/fill14.py) evaluation on A6000 (CPU, nice 19).
#   bash fill14_eval.sh val600            : fill + REAL Qwen re-extraction of the filled text (4 groups) + merge scoring
#                                           + merge-vs-real agreement (fill14_agree.py)
#   bash fill14_eval.sh valfull [groups]  : fill + merge scoring only (needs <g>.eval_full_ok: the original full-val
#                                           extraction is reused; the merge was validated on val600 against real Qwen)
# gen  : m4/fill14/gen/<g>/<split>.{jsonl,merge_ext.jsonl,summary.json}
# out  : m4/eval/results_fill14/<g>_<split>_fill{,_merge}/ ; agreement m4/eval/results_fill14/agree_<split>.json
# Qwen : tunnel A6000:18014 -> Spark 127.0.0.1:8003 (own tunnel; eval_full.sh uses 18013), jobs FILL_JOBS (16);
#        shared cache m4/eval/qwen_extract_cache.jsonl
# Launch: setsid nohup bash queue_cue/fill14_eval.sh val600 > queue_cue/logs/fill14_val600.out 2>&1 < /dev/null &
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
Q=$M4/queue_cue
PY=$TONY/tools/vllm-env/bin/python
PORT=${FILL_PORT:-18014}
JOBS=${FILL_JOBS:-16}
SPLIT=${1:?val600|valfull}
shift
GROUPS_=${*:-"U_oof U_gt T_oof T_gt"}
OUT=$M4/eval/results_fill14
log() { echo "[$(date '+%m-%d %H:%M:%S')] [fill14] $*" | tee -a "$Q/STATUS.log"; }
mkdir -p "$Q/logs" "$OUT"

(
    while true; do
        ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
            -L $PORT:127.0.0.1:8003 daikso_spark@192.168.0.95 < /dev/null
        sleep 15
    done
) &
KEEPER=$!
cleanup() { local c; c=$(pgrep -P $KEEPER); kill $KEEPER 2>/dev/null; [ -n "$c" ] && kill $c 2>/dev/null; true; }
trap cleanup EXIT
endpoint_ok() { $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10)" 2>/dev/null; }
for i in $(seq 1 120); do endpoint_ok && break; sleep 30; done

ev() {  # name gen [extra args...]
    local name=$1 gen=$2; shift 2
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 $PY $M4/code_v2/eval/evaluate.py --gen "$gen" --name "$name" \
        --labels $TONY/VLM/OWL/data/labels_disease14_all.jsonl --meta $TONY/VLM/merlin_disease14/metadata.csv \
        --disease14-dir $TONY/VLM/merlin_disease14 --targets $M4/inputs/targets_v2/targets_val.jsonl --split val \
        --url http://127.0.0.1:$PORT/v1 --jobs "$JOBS" --cache $M4/eval/qwen_extract_cache.jsonl --out-root "$OUT" \
        --bertscore-model $TONY/models/roberta-large --bertscore-threads 4 "$@" >> "$Q/logs/fill14_${name}.log" 2>&1 < /dev/null
}

if [ "$SPLIT" = val600 ]; then src=val600; else src=val_full; fi
log "start $SPLIT: $GROUPS_ (jobs $JOBS)"
for g in $GROUPS_; do
    orig=$M4/gen_cue/$g/$src.jsonl
    if [ "$SPLIT" = val600 ]; then pc=$M4/eval/results_cue/${g}_val600/per_case_categories.jsonl
    else
        [ -e "$Q/state/$g.eval_full_ok" ] || { log "$g: original full-val evaluation not done -> skip"; continue; }
        pc=$M4/eval/results_cue/${g}_valfull/per_case_categories.jsonl
    fi
    fo=$M4/fill14/gen/$g/$SPLIT
    $PY $M4/code_v2/fill14.py --gen "$orig" --cues "${orig%.jsonl}.cues.jsonl" --per-case "$pc" \
        --bank $M4/fill14/inputs/sentence_bank.json --measure $TONY/VLM/OWL/report_gen/oof/out/measure_trval.csv \
        --calib $M4/fill14/inputs/measure_calibration.json --out "$fo" >> "$Q/logs/fill14_${g}_${SPLIT}.log" 2>&1 \
        || { log "$g: fill14.py failed"; continue; }
    # merge scoring: original extraction (cached) + template state of the filled findings
    ev "${g}_${SPLIT}_fill_merge" "$fo.jsonl" --extract-from "$orig" --ext-override "$fo.merge_ext.jsonl" --no-request \
        --cue-log "${orig%.jsonl}.cues.jsonl"
    log "$g $SPLIT merge rc $?"
    if [ "$SPLIT" = val600 ]; then
        for pass in 1 2; do
            ev "${g}_${SPLIT}_fill" "$fo.jsonl" --cue-log "${orig%.jsonl}.cues.jsonl"
            rc=$?
            nf=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['qwen']['n_failed_or_unlabelled'])" \
                 "$OUT/${g}_${SPLIT}_fill/metrics.json" 2>/dev/null || echo NA)
            log "$g $SPLIT real re-extraction pass $pass rc $rc, failed $nf"
            [ $rc = 0 ] && [ "$nf" = 0 ] && break
        done
    fi
done
if [ "$SPLIT" = val600 ]; then
    $PY $M4/code_v2/fill14_agree.py --root "$OUT" --split val600 --gen-root $M4/fill14/gen $GROUPS_ \
        > "$OUT/agree_val600.json" 2>> "$Q/logs/fill14_agree.log"
    log "agreement -> $OUT/agree_val600.json"
fi
log "finished $SPLIT"
