#!/usr/bin/env python3
"""Short GPU benchmark of method 4 with the real backbone and SYNTHETIC data (no report text): a few optimizer
steps of stage 1, stage 2 (LoRA) and text-only stage 2, plus a short greedy generation; reports peak GPU memory and
seconds per micro-step, from which the full-run time is estimated. Not a training run (nothing is saved).

usage (A6000): python gpu_smoke.py --llm ~/Desktop/Tony/models/Lingshu-7B --meta <features>/token_meta.csv \
                   --work <scratch dir> [--backbone lingshu7b --steps 6 --micro-bs 4]
"""
import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_jsonl import make_prompt  # noqa: E402
from m4_common import DISEASES  # noqa: E402
from m4_data import M4Dataset, collate  # noqa: E402
from train import build  # noqa: E402

LINES = [  # generic sentences written for this benchmark (not from any report); ~150 target tokens per case
    "[hepatic_steatosis] Liver and biliary tree: The liver is normal in size with homogeneous attenuation.",
    "[biliary_ductal_dilation] No intrahepatic or extrahepatic biliary ductal dilatation.",
    "[gallstones,surgically_absent_gallbladder] Gallbladder: Normal appearance without radiopaque stones.",
    "[splenomegaly] Spleen: Normal in size.",
    "[hydronephrosis] Kidneys and ureters: No hydronephrosis or perinephric fluid.",
    "[bowel_obstruction,appendicitis] Gastrointestinal tract: No bowel obstruction. The appendix is normal.",
    "[ascites,free_air] Peritoneal cavity: No free fluid or free air.",
    "[abdominal_aortic_aneurysm,atherosclerosis] Vasculature: Moderate atherosclerotic calcification without aneurysm.",
    "[osteopenia] Diffuse osteopenia.",
    "[fracture] Mild compression deformity of a lower thoracic vertebral body, age indeterminate.",
]


def synth(work, meta_csv, n):
    import csv
    T = sum(1 for _ in open(meta_csv)) - 1
    (work / "feat").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        cid = f"SYN{i:04d}"
        mask = rng.random(T) > 0.07
        x = rng.standard_normal((T, 448)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) / 1.732
        np.savez(work / "feat" / f"{cid}.npz", tok=(x * mask[:, None]).astype(np.float16), mask=mask)
        probs = {d: float(rng.random()) for d in DISEASES}
        meta = {"phase": "portal_venous", "manufacturer": "GE MEDICAL SYSTEMS", "manufacturermodelname": "LightSpeed VCT"}
        meas = {"abd_eq_rmax": "19.0", "sp_cc_main_cm": "10.0", "sp_ml": "200"}
        calib = {"aaa_cm": {"col": "abd_eq_rmax", "a": 1.18, "b": -0.3}, "spleen_cm": {"col": "sp_cc_main_cm", "a": 0.7, "b": 4.9}}
        rows.append({"case_id": cid, "split": "train", "feature_path": str(work / "feat" / f"{cid}.npz"),
                     "prompt": make_prompt(meta, probs, meas, calib), "target": "\n".join(LINES),
                     "target_diseases": DISEASES})
    (work / "syn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return work / "syn.jsonl"


def bench(tag, args, stage, text_only, data, steps, init_adapter=""):
    ns = argparse.Namespace(llm=args.llm, backbone=args.backbone, meta=args.meta, text_only=text_only, stage=stage,
                            init_adapter=init_adapter, res_dim=1024, res_depth=2, res_heads=8, lora_r=16,
                            lora_alpha=32, lora_dropout=0.05, grad_ckpt=True, dtype="bf16")
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tk, model = build(ns, dev)
    load_s = time.time() - t0
    ds = M4Dataset(data, tk, not text_only, 1024, backbone=args.backbone)
    col = functools.partial(collate, pad_id=tk.pad_token_id, train=True)
    groups = []
    if model.adapter is not None:
        groups.append({"params": list(model.adapter.parameters()), "lr": 1e-3 if stage == 1 else 1e-4})
    if stage == 2:
        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": 2e-4})
    opt = torch.optim.AdamW(groups)
    times, losses = [], []
    lens = None
    model.train()
    for s in range(steps):
        b = col([ds[(s * args.micro_bs + j) % len(ds)] for j in range(args.micro_bs)])
        lens = int(b["attention_mask"].sum(1).float().mean())
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["labels"].to(dev),
                      b["tok"].to(dev) if "tok" in b else None, b["mask"].to(dev) if "mask" in b else None)
        o.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        losses.append(round(o.loss.item(), 3))
    res = {"tag": tag, "load_sec": round(load_s, 1), "seq_len_mean": lens, "micro_bs": args.micro_bs,
           "sec_per_micro_step": round(float(np.median(times[1:])), 3), "first_step_sec": round(times[0], 2),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2), "losses": losses,
           "trainable_m": round(sum(p.numel() for g in groups for p in g["params"]) / 1e6, 1)}
    if stage == 2 and not text_only:          # short generation timing (batch 16, 64 new tokens)
        model.eval()
        model.llm.config.use_cache = True
        dsg = M4Dataset(data, tk, True, backbone=args.backbone, train=False)
        tk.padding_side = "left"
        bg = collate([dsg[j] for j in range(16)], tk.pad_token_id, train=False)
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(bg["input_ids"].to(dev), bg["attention_mask"].to(dev), bg["tok"].to(dev),
                               bg["mask"].to(dev), max_new_tokens=64, min_new_tokens=64,
                               pad_token_id=tk.pad_token_id)
        torch.cuda.synchronize()
        res["gen_b16_64tok_sec"] = round(time.time() - t, 2)
        res["gen_shape"] = list(g.shape)
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--backbone", default="lingshu7b")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=15309)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    data = synth(work, a.meta, 32)
    out = [bench("stage1", a, 1, False, data, a.steps), bench("stage2", a, 2, False, data, a.steps),
           bench("text_only_stage2", a, 2, True, data, a.steps)]
    steps_ep = a.n_train / a.micro_bs
    for r in out:
        r["est_hours_per_epoch"] = round(r["sec_per_micro_step"] * steps_ep / 3600, 2)
    print(json.dumps(out, indent=1))
    (work / "gpu_smoke_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
