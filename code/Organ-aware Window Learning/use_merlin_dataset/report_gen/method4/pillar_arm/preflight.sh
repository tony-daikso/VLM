#!/bin/bash
# Option B (arm P_gt) preflight on A6000 (own dirs only: m4/code_p, m4/p_arm/preflight; never the peer's improve/state).
#   1. preflight data subset: data_v2_nounit train rows that have Pillar features in the store (train 80; 16 more train
#      rows play "val" when no val features exist yet -- plumbing only)
#   2. unit tests (core + data on the real tokenizer / rows / store)
#   3. CPU end to end with a tiny random LLM: train_p stage 1 (2 steps) -> stage 2 (2 steps, init from stage 1) ->
#      infer_p (4 rows), P_gt switches (gt_dropout sample 0.3, unclear oof, meas-dropout 0); + global / unit ablations
#   4. GPU (only if >= 20 GB VRAM free AND >= 25 GB RAM available; capped with --gpu-mem-cap-gb): the same with the
#      Qwen3-1.7B stand-in at the real micro-bs 4 / max-len 1280 -> peak memory, s/step
#   5. writes m4/code_p/state/pre_P_gt.ok (summary) or pre_P_gt.failed
# usage: bash preflight.sh <store dir> [gpu|cpu]
set -uo pipefail
TONY=/home/advantech/Desktop/Tony
M4=$TONY/VLM/OWL/report_gen/m4
P=$M4/code_p
W=$M4/p_arm/preflight
STORE=$1
MODE=${2:-gpu}
PY=$TONY/tools/vllm-env/bin/python
export LD_LIBRARY_PATH=$TONY/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TONY/tools/m4_pylib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W $P/state $P/logs
rm -f $P/state/pre_P_gt.ok $P/state/pre_P_gt.failed
fail() { echo "FAILED: $*"; echo "$*" > $P/state/pre_P_gt.failed; exit 1; }
cd $P/pillar_arm

echo "== 1. data subset"
$PY - "$STORE" "$W/data" <<'EOF' || fail "subset"
import json, sys
from pathlib import Path
store, out = Path(sys.argv[1]), Path(sys.argv[2])
import numpy as np
ids = (store / "ids.txt").read_text().split()
done = np.load(store / "done.npy", mmap_mode="r")
have = {c for c, d in zip(ids, done) if d}
D = Path("/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/improve/data_v2_nounit")
tr = [l for l in open(D / "train.jsonl") if json.loads(l)["case_id"] in have]
va = [l for l in open(D / "val.jsonl") if json.loads(l)["case_id"] in have]
if len(va) < 8:                                   # no val features yet: 16 held-out train rows play val (plumbing)
    va, tr = tr[-16:], tr[:-16]
out.mkdir(parents=True, exist_ok=True)
(out / "train.jsonl").write_text("".join(tr[:80]))
(out / "val.jsonl").write_text("".join(va[:16]))
print(f"train {len(tr[:80])} val {len(va[:16])}")
EOF

echo "== 2. unit tests"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY test_p_arm.py core > $W/test_core.json || fail "test core"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY test_p_arm.py data --store $STORE --data $W/data > $W/test_data.json 2> $W/test_data.err || fail "test data"
tail -n +1 $W/test_core.json $W/test_data.json

echo "== 3. CPU tiny LLM end to end"
if [ ! -d $W/tiny/tiny_plain ]; then
  CUDA_VISIBLE_DEVICES= $PY -c "
import sys; sys.path.insert(0, '$P'); from pathlib import Path
import smoke_test; w = Path('$W/tiny'); w.mkdir(parents=True, exist_ok=True)
smoke_test.write_ckpts('$TONY/models/Lingshu-7B', w)" || fail "tiny ckpt"
fi
GT="--cue-source gt_dropout --cue-gt $M4/cue/cue_gt.json --cue-gt-map sample --cue-dropout 0.3 --cue-gt-unclear oof --meas-dropout 0"
CPU="--llm $W/tiny/tiny_plain --backbone tiny_random --device cpu --dtype fp32 --micro-bs 2 --grad-accum 1 --max-steps 2 --num-workers 0 --res-dim 32 --res-heads 4 --log-every 1 --max-len 1280 --val-n 4"
for m in both global unit; do
  R=$W/cpu_$m
  rm -rf $R
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY train_p.py --pillar-root $STORE --pillar-mode $m --stage 1 --data $W/data --cue-source oof $CPU --out $R/s1 > $W/cpu_${m}_s1.log 2>&1 || fail "cpu $m s1 (see $W/cpu_${m}_s1.log)"
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY train_p.py --pillar-root $STORE --pillar-mode $m --stage 2 --data $W/data $GT $CPU --init-adapter $R/s1/best/adapter.pt --out $R/s2 > $W/cpu_${m}_s2.log 2>&1 || fail "cpu $m s2 (see $W/cpu_${m}_s2.log)"
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY infer_p.py --llm $W/tiny/tiny_plain --run $R/s2/final --data $W/data/val.jsonl --out $R/gen4.jsonl --device cpu --dtype fp32 --limit 4 --batch 2 --num-workers 0 --max-new-tokens 8 --cue-gt $M4/cue/cue_gt.json > $W/cpu_${m}_infer.log 2>&1 || fail "cpu $m infer (see $W/cpu_${m}_infer.log)"
  [ "$(wc -l < $R/gen4.jsonl)" = 4 ] || fail "cpu $m gen rows"
  grep -h "step 2/2\|val loss\|pillar\]" $W/cpu_${m}_s1.log $W/cpu_${m}_s2.log | tr '\n' ' '; echo
done

GPU_NOTE="skipped"
if [ "$MODE" = gpu ]; then
  echo "== 4. GPU stand-in (Qwen3-1.7B)"
  vfree=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%d", ($2-$1)/1024}')
  rfree=$(free -g | awk '/^Mem:/{print $7}')
  echo "VRAM free ${vfree} GB, RAM available ${rfree} GB"
  if [ "$vfree" -ge 20 ] && [ "$rfree" -ge 25 ]; then
    G="--llm $TONY/models/Qwen3-1.7B --backbone qwen3_1p7b --micro-bs 4 --grad-accum 1 --max-steps 2 --num-workers 2 --max-len 1280 --val-n 8 --log-every 1 --min-free-gb 18 --gpu-mem-cap-gb 14"
    R=$W/gpu_both; rm -rf $R
    nice -n 5 $PY train_p.py --pillar-root $STORE --stage 1 --data $W/data --cue-source oof $G --out $R/s1 > $W/gpu_s1.log 2>&1 || fail "gpu s1 (see $W/gpu_s1.log)"
    nice -n 5 $PY train_p.py --pillar-root $STORE --stage 2 --data $W/data $GT $G --init-adapter $R/s1/best/adapter.pt --out $R/s2 > $W/gpu_s2.log 2>&1 || fail "gpu s2 (see $W/gpu_s2.log)"
    nice -n 5 $PY infer_p.py --llm $TONY/models/Qwen3-1.7B --run $R/s2/final --data $W/data/val.jsonl --out $R/gen4.jsonl --limit 4 --batch 4 --num-workers 0 --max-new-tokens 64 --min-free-gb 18 --cue-gt $M4/cue/cue_gt.json > $W/gpu_infer.log 2>&1 || fail "gpu infer (see $W/gpu_infer.log)"
    # stage-1 dynamics with 308 tokens: 40 steps, warmup 0.7 (= the real run's 28 warmup steps), loss must go down
    nice -n 5 $PY train_p.py --pillar-root $STORE --stage 1 --data $W/data --cue-source oof ${G/--max-steps 2/--max-steps 40} --epochs 2 --warmup 0.7 --out $R/dyn > $W/gpu_dyn.log 2>&1 || fail "gpu dyn (see $W/gpu_dyn.log)"
    grep -h "step \|val loss\|peak\|trainable" $W/gpu_s1.log $W/gpu_s2.log $W/gpu_dyn.log
    GPU_NOTE="qwen3_1p7b ok"
  else
    GPU_NOTE="skipped (VRAM ${vfree} GB / RAM ${rfree} GB free)"
  fi
fi
$PY - "$W" "$GPU_NOTE" "$STORE" > $P/state/pre_P_gt.tmp <<'EOF' || fail "summary"
import json, sys, time
from pathlib import Path
w = Path(sys.argv[1])


def js(f):   # test output = progress lines, then one JSON object
    t = (w / f).read_text()
    return json.loads(t[t.index("{"):] if t.startswith("{") else t[t.index("\n{") + 1:])


peak = [l.strip() for f in ("gpu_s1.log", "gpu_s2.log") if (w / f).exists() for l in open(w / f) if "peak GPU" in l]
print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "store": sys.argv[3], "gpu": sys.argv[2], "gpu_peak": peak,
                  "core": js("test_core.json"), "data": js("test_data.json")}, indent=1))
EOF
mv $P/state/pre_P_gt.tmp $P/state/pre_P_gt.ok
echo "PASS -> $P/state/pre_P_gt.ok ($GPU_NOTE)"
