#!/usr/bin/env python3
"""Option B (arm P_gt) training = method-4 train.py with the Pillar adapter / dataset patched in (p_arm.patch).
All train.py switches work unchanged (--cue-source gt_dropout --cue-gt-map sample --cue-dropout 0.3
--cue-gt-unclear oof --meas-dropout 0 ..., LoRA r16/a32, stage 1 / 2, saving every epoch + best + final).
Extra switches (consumed here, the rest goes to train.py):
  --pillar-root DIR          pack_pillar.py store (ids.txt, grid16.npy, units.npy, uinfo.npy, zvalid.npy, done.npy)
  --pillar-mode both|global|unit   (P_gt = both; global / unit = ablations)
  --pillar-n-global 256
  --pillar-strict            refuse rows without Pillar features (default: all-missing image, count printed)
  --gpu-mem-cap-gb X         preflight only: torch per-process memory cap (this process OOMs before it can squeeze a
                             co-running job)
Each saved dir (epochN / best / final) also gets pillar.json (mode, token count/order, unit list, store md5) -> infer_p.py.
Stage 2 with --init-adapter checks that the stage-1 pillar.json has the same mode / n_global.
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
    ap.add_argument("--pillar-root", required=True)
    ap.add_argument("--pillar-mode", default="both", choices=p_arm.MODES)
    ap.add_argument("--pillar-n-global", type=int, default=256)
    ap.add_argument("--pillar-strict", action="store_true")
    ap.add_argument("--gpu-mem-cap-gb", type=float, default=0)
    pa, rest = ap.parse_known_args()
    if "--text-only" in rest:
        sys.exit("P arm always has image tokens (use train.py for text-only)")
    for bad in ("--lr-flip-prob", "--aux-unit-loss", "--resampler-groups", "--unit-labels-dir"):
        if bad in rest:
            sys.exit(f"{bad} is an OWL-token option, not available in the P arm")
    p_arm.CFG.root, p_arm.CFG.mode, p_arm.CFG.n_global = pa.pillar_root, pa.pillar_mode, pa.pillar_n_global
    p_arm.CFG.allow_missing = not pa.pillar_strict
    if "--init-adapter" in rest:
        init = Path(rest[rest.index("--init-adapter") + 1])
        pj = init.parent / "pillar.json"
        if not pj.exists():
            sys.exit(f"--init-adapter {init}: no pillar.json next to it (not a P-arm adapter)")
        c = json.loads(pj.read_text())
        if (c["mode"], c["n_global"]) != (pa.pillar_mode, pa.pillar_n_global):
            sys.exit(f"stage-1 adapter is mode {c['mode']} / n_global {c['n_global']}")
    if pa.gpu_mem_cap_gb:
        import torch
        if torch.cuda.is_available():
            tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1.0, pa.gpu_mem_cap_gb / tot), 0)
            print(f"[pillar] GPU memory cap {pa.gpu_mem_cap_gb} GB of {tot:.1f}", flush=True)
    import train
    p_arm.patch(train)
    orig_save = train.save

    def save(model, args, out_dir, extra):
        orig_save(model, args, out_dir, {**extra, "pillar": p_arm.CFG.to_json()})
        p_arm.write_cfg(out_dir)
    train.save = save
    print(f"[pillar] mode {pa.pillar_mode}, {p_arm.CFG.n_tokens} image tokens, store {pa.pillar_root}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    try:
        train.main()
    finally:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0:
                print(f"[pillar] peak GPU memory allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB, "
                      f"reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB", flush=True)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
