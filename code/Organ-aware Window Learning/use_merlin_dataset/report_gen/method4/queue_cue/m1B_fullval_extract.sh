#!/bin/bash
# Method-1 1B (template, t_f1) full-val (5,055) Qwen re-extraction, run ON Spark against its own merlin-qwen-fp8 (:8003)
# (2026-11-12). Same evaluate.py (md5 64bc2767...), same labels / metadata / disease14 prompt + run_disease14.py as the
# peer's eval_full2.sh for U_gt / T_gt, so the full-val comparison is apples-to-apples.
#   Method 1 is a template baseline (classifier score + t_f1 threshold + train sentence bank): there is no neural
#   generation and no GPU step; the full-val drafts were assembled on 11-09 by method1/m1_assemble.py
#   (val_drafts_B_text.jsonl, 5,055 rows, the same file that gave val600 macro F1 0.665).
#   (1) full val: cache seeded with the 11-09 extractions of the 600-case subset (same weights / prompt / max_tokens /
#       user message -> same cache key, exactly as a shared cache would be used); the other 4,455 are requested.
#   (2) then a FRESH re-extraction of the 600 subset into a separate cache (re-extraction noise check vs 11-09).
#   Up to 3 passes per stage (later passes only retry failed extractions).
#   Scoring with BERTScore runs on the CT-rate box (m1B_fullval_score.sh) from a copy of the caches.
#   Concurrency JOBS (16): the endpoint is shared with the peer's evals (max-num-seqs 64).
#   Never touches :8005 (std14-qwen) or the A6000. Test split is never read (evaluate.py drops it).
# usage (Spark): setsid nohup bash m1B_fullval_extract.sh > logs/extract.out 2>&1 < /dev/null &   stop: touch logs/STOP
set -uo pipefail
W=${W:-$HOME/Desktop/Tony/VLM/report_gen_m1_fullval}
PY=${PY:-$HOME/Desktop/Tony/radar_env/bin/python}
JOBS=${EVAL_JOBS:-16}
URL=http://127.0.0.1:8003/v1
GEN=$W/inputs/val_drafts_B_text.jsonl
log() { echo "[$(date '+%m-%d %H:%M:%S')] [m1B] $*" | tee -a "$W/logs/STATUS.log"; }

ev() {  # name cache extra-args...
    local n=$1 c=$2; shift 2
    OMP_NUM_THREADS=2 nice -n 10 $PY $W/code/evaluate.py --gen "$GEN" --name "$n" \
        --labels $W/inputs/labels_disease14_all.jsonl --meta $W/inputs/metadata.csv --disease14-dir $W/disease14 \
        --targets $W/inputs/targets_val.jsonl --split val --url $URL --jobs "$JOBS" \
        --cache "$c" --out-root $W/eval/results "$@" < /dev/null
}
nfail() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['n_failed_or_unlabelled'])" \
          "$W/eval/results/$1/extract_status.json" 2>/dev/null || echo NA; }

echo $$ > "$W/logs/extract.pid"
[ -s $W/eval/qwen_extract_cache.jsonl ] || cp $W/eval/seed_cache_m1B_1109.jsonl $W/eval/qwen_extract_cache.jsonl
log "started (pid $$), jobs $JOBS"
for stage in full recheck; do
    [ -e "$W/logs/STOP" ] && { log "STOP -> exit"; exit 0; }
    if [ $stage = full ]; then name=m1B_valfull; cache=$W/eval/qwen_extract_cache.jsonl; extra=()
    else name=m1B_sub600_recheck; cache=$W/eval/qwen_extract_cache_recheck600.jsonl
         extra=(--ids $W/inputs/val_qwen_subset_ids.txt); fi
    for pass in 1 2 3; do
        log "$name: extract pass $pass"
        ev "$name" "$cache" "${extra[@]}" --extract-only >> "$W/logs/extract_$name.log" 2>&1
        nf=$(nfail "$name"); log "$name: pass $pass done, failed/unlabelled $nf"
        [ "$nf" = 0 ] && break
        [ -e "$W/logs/STOP" ] && break
        sleep 60
    done
    touch "$W/logs/$name.extract_done"
done
log "extraction finished"; touch "$W/logs/extract.finished"
