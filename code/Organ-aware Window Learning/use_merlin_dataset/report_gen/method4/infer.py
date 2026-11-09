#!/usr/bin/env python3
"""Method 4 inference: saved run (adapter.pt + lora/) -> {case_id, findings: {disease: sentence}, raw} jsonl,
the input format of report_gen/eval/evaluate.py.

usage: python infer.py --llm <LLM dir> --run runs/m4_s2/best --data <jsonl dir>/val.jsonl --out gen_val.jsonl
       (--text-only for the ablation run; the test split is refused unless --allow-test after the user unlocks it)
error-cue test (val only): --cue-gt <cue_gt.json> --cue-flip-pos2neg 0.5 --cue-flip-neg2pos 0.05 [--cue-flip-seed N]
       flips part of the classifier cues before generation (m4_common.flip_cues; per-case RNG seeded by seed + case_id,
       so the flips do not depend on batch order or --limit) and writes <out>.cues.jsonl
       ({case_id, cue: original cue, flips: {disease: [old, new, direction]}}) + <out>.cues.meta.json (thresholds,
       probabilities, seed). Give --cue-gt alone (no flip probabilities) to log the original cues of a normal run.
       evaluate.py --cue-log <out>.cues.jsonl then scores "cue wrong but report still consistent" per class.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m4_common import BACKBONES, N_IMG_TOKENS, flip_cues, parse_cue, parse_output, replace_cue  # noqa: E402
from m4_data import M4Dataset, collate  # noqa: E402
from m4_model import ImageAdapter, ReportGenModel, read_token_meta  # noqa: E402
from train import check_gpu, load_llm  # noqa: E402


def apply_cue_flips(rows, cue_gt_path, p_pos2neg, p_neg2pos, seed, log_path):
    """error-cue test: flip classifier cues of val rows in place (prompt line rewritten in the same format) and log
    original cues + flips. Deterministic per case: random.Random(f"{seed}:{case_id}")."""
    z = json.load(open(cue_gt_path))
    thr, pools = z["thresholds"], z["pools"]
    bad = [r["case_id"] for r in rows if r["split"] != "val"]
    if bad and (p_pos2neg or p_neg2pos):
        raise SystemExit(f"the error-cue test runs on val only; {len(bad)} other rows (e.g. {bad[0]})")
    n_flip = {"pos2neg": 0, "neg2pos": 0}
    with open(log_path, "w") as f:
        for r in rows:
            cue, _ = parse_cue(r["prompt"])
            new, flips = flip_cues(cue, thr, pools, p_pos2neg, p_neg2pos, random.Random(f"{seed}:{r['case_id']}"))
            if flips:
                r["prompt"] = replace_cue(r["prompt"], new)
            for v in flips.values():
                n_flip[v[2]] += 1
            f.write(json.dumps({"case_id": r["case_id"], "cue": cue, "flips": flips}) + "\n")
    meta = {"thresholds": thr, "p_pos2neg": p_pos2neg, "p_neg2pos": p_neg2pos, "seed": seed, "n_cases": len(rows),
            "n_flips": n_flip, "cue_gt": str(cue_gt_path),
            "cue_gt_md5": hashlib.md5(Path(cue_gt_path).read_bytes()).hexdigest()}
    Path(str(log_path)[: -len(".jsonl")] + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[cue] {n_flip} flips over {len(rows)} cases -> {log_path}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--run", required=True, help="saved dir with manifest.json, adapter.pt and/or lora/")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--meta", default="")
    ap.add_argument("--feat-root", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--cue-gt", default="", help="build_cue_gt.py cue_gt.json (thresholds + train OOF pools); "
                                                 "with it the original cues (and flips) are logged to <out>.cues.jsonl")
    ap.add_argument("--cue-flip-pos2neg", type=float, default=0.0,
                    help="error-cue test: probability that a cue at/above the method-1 threshold is made negative")
    ap.add_argument("--cue-flip-neg2pos", type=float, default=0.0,
                    help="error-cue test: probability that a cue below the threshold is made positive")
    ap.add_argument("--cue-flip-seed", type=int, default=20261109)
    a = ap.parse_args()
    if (a.cue_flip_pos2neg or a.cue_flip_neg2pos) and not a.cue_gt:
        sys.exit("--cue-flip-* needs --cue-gt")

    run = Path(a.run)
    man = json.loads((run / "manifest.json").read_text())
    margs = man["args"]
    assert bool(margs.get("text_only")) == a.text_only, "text-only flag differs from the trained run"
    device = torch.device(a.device)
    if device.type == "cuda":
        check_gpu(a.min_free_gb, a.force_gpu)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    bbn = margs.get("backbone", "lingshu7b")
    tk, llm = load_llm(a.llm, dtype, device, bbn)
    if (run / "lora").exists():
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, run / "lora").merge_and_unload()
    llm.eval()
    llm.config.use_cache = True
    adapter = None
    if not a.text_only:
        meta = read_token_meta(a.meta or margs["meta"])
        adapter = ImageAdapter(meta, llm.config.hidden_size, dim=margs["res_dim"], n_latents=N_IMG_TOKENS,
                               depth=margs["res_depth"], heads=margs["res_heads"],
                               aux_unit=margs.get("aux_unit_loss", 0) > 0,
                               groups=margs.get("resampler_groups", False)).to(device)
        adapter.load_state_dict(torch.load(run / "adapter.pt", map_location=device))
        adapter.eval()
    model = ReportGenModel(llm, adapter, tk.convert_tokens_to_ids(BACKBONES[bbn]["img_token"]) if adapter else None)
    tk.padding_side = "left"
    ds = M4Dataset(a.data, tk, not a.text_only, limit=a.limit, feat_root=a.feat_root or None,
                   backbone=bbn, allow_test=a.allow_test, train=False)
    if a.cue_gt:
        apply_cue_flips(ds.rows, a.cue_gt, a.cue_flip_pos2neg, a.cue_flip_neg2pos, a.cue_flip_seed,
                        str(Path(a.out).with_suffix("")) + ".cues.jsonl")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.num_workers, pin_memory=False,
                    collate_fn=functools.partial(collate, pad_id=tk.pad_token_id, train=False))
    end = tk.convert_tokens_to_ids("<|im_end|>")
    n = 0
    with open(a.out, "w") as f:
        for b in dl:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and a.dtype == "bf16"):
                gen = model.generate(b["input_ids"].to(device), b["attention_mask"].to(device),
                                     b["tok"].to(device) if "tok" in b else None,
                                     b["mask"].to(device) if "mask" in b else None,
                                     max_new_tokens=a.max_new_tokens, eos_token_id=end, pad_token_id=tk.pad_token_id)
            for cid, g in zip(b["case_id"], gen):
                text = tk.decode(g, skip_special_tokens=True).strip()
                f.write(json.dumps({"case_id": cid, "findings": parse_output(text), "raw": text,
                                    "system": str(run)}, ensure_ascii=False) + "\n")
                n += 1
            print(f"{n}/{len(ds)}", flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
