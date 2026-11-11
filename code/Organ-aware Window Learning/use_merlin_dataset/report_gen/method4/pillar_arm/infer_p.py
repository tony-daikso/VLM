#!/usr/bin/env python3
"""Option B (arm P_gt) inference = method-4 infer.py with the Pillar adapter / dataset patched in; the adapter layout
(mode, n_global) comes from <run>/pillar.json, the store from --pillar-root (default: the one used in training).
All infer.py switches work unchanged (--cue-gt logging, --cue-flip-*, --limit, --batch; test refused).
usage: python infer_p.py --llm <Lingshu-7B> --run runs/P_gt/best --data improve/data/val600_v2_nounit.jsonl \
          --out gen_p/P_gt_val600.jsonl --batch 32 --max-new-tokens 512 --cue-gt cue/cue_gt.json [--pillar-root DIR]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import p_arm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pillar-root", default="")
    ap.add_argument("--pillar-strict", action="store_true")
    ap.add_argument("--run", required=True)
    pa, rest = ap.parse_known_args()
    if "--text-only" in rest:
        sys.exit("P arm always has image tokens")
    c = json.loads((Path(pa.run) / "pillar.json").read_text())
    p_arm.CFG.root = pa.pillar_root or c["root"]
    p_arm.CFG.mode, p_arm.CFG.n_global = c["mode"], c["n_global"]
    p_arm.CFG.allow_missing = not pa.pillar_strict
    assert p_arm.CFG.n_tokens == c["n_tokens"], "token count differs from the trained run"
    import infer
    p_arm.patch(infer)
    print(f"[pillar] mode {c['mode']}, {c['n_tokens']} image tokens, store {p_arm.CFG.root}", flush=True)
    sys.argv = [sys.argv[0], "--run", pa.run] + rest
    infer.main()


if __name__ == "__main__":
    main()
