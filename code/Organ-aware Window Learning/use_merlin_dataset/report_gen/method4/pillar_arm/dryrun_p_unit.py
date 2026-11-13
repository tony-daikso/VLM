#!/usr/bin/env python3
"""Arm P_unit_gt_pil (= P_gt image tokens + the A+ text of T_unit_gt_pil): CPU-only data dry run, no model, no GPU.

Uses the same classes as the training job (p_arm.PDataset on top of m4_data.M4Dataset, CueGT, UnitCueGT, collate) with
the real tokenizer, and prints aggregates only (no report text, no case ids):
  1. join: rows of <data>/train.jsonl, val.jsonl (and --val600) without Pillar features; case-id set / target equality
     with --ref-data (the P_gt text data);
  2. token length of prompt (image tokens included) + target (+ <|im_end|>) over --passes resamplings of the gt cue
     lines, percentiles, and the number of train rows whose target would be cut at each --max-len value;
  3. --batches collated batches at --use-max-len with the real Pillar features: tensor shapes, image-token count, and
     that the label part decodes back to the full target.
usage (A6000, next to train_p.py; parent dir holds train.py / m4_data.py):
  CUDA_VISIBLE_DEVICES= python dryrun_p_unit.py --llm <Lingshu-7B> --pillar-root <pillar_feats> \
      --data improve/data_unit_pil --unit-cue-gt improve/data_unit_pil/unit_cue_pil.json --cue-gt cue/cue_gt.json \
      --ref-data improve/data_v2_nounit --val600 improve/data/val600_unit_pil.jsonl --max-len 1024 1280 1536
"""
import argparse
import functools
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import p_arm  # noqa: E402
from m4_common import BACKBONES  # noqa: E402
from m4_data import CueGT, UnitCueGT, collate  # noqa: E402

PCT = (50, 90, 95, 99, 99.9, 100)


def pct(x):
    return " ".join(f"p{p}={int(np.percentile(x, p))}" for p in PCT)


class LenDS(p_arm.PDataset):
    """same items, features skipped (length pass only)."""

    def features(self, r):
        return torch.zeros(1), torch.zeros(1)


def ids_targets(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["case_id"]] = r.get("target", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--pillar-root", required=True)
    ap.add_argument("--pillar-mode", default="both")
    ap.add_argument("--pillar-n-global", type=int, default=256)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref-data", default="")
    ap.add_argument("--val600", default="")
    ap.add_argument("--cue-gt", required=True)
    ap.add_argument("--unit-cue-gt", default="")
    ap.add_argument("--unit-cue-grades", action="store_true")
    ap.add_argument("--cue-dropout", type=float, default=0.3)
    ap.add_argument("--unit-cue-dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, nargs="+", default=[1024, 1280, 1536])
    ap.add_argument("--use-max-len", type=int, default=1536)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--backbone", default="lingshu7b")
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)

    from transformers import AutoTokenizer
    bb = BACKBONES[a.backbone]
    tk = AutoTokenizer.from_pretrained(a.llm, trust_remote_code=bb["trust_remote_code"])
    missing = [t for t in (bb["img_token"], bb["img_start"], bb["img_end"])
               if tk.convert_tokens_to_ids(t) in (None, tk.unk_token_id)]
    print(f"[tok] special image tokens missing from the tokenizer: {missing}")
    if tk.pad_token_id is None:
        tk.pad_token = "<|endoftext|>" if "<|endoftext|>" in tk.get_vocab() else tk.eos_token
    p_arm.CFG.root, p_arm.CFG.mode, p_arm.CFG.n_global = a.pillar_root, a.pillar_mode, a.pillar_n_global
    p_arm.CFG.allow_missing = False
    n_img = p_arm.CFG.n_tokens
    print(f"[pillar] {n_img} image tokens")
    data = Path(a.data)

    # 1. join / reference
    if a.ref_data:
        for sp in ("train", "val"):
            new, ref = ids_targets(data / f"{sp}.jsonl"), ids_targets(Path(a.ref_data) / f"{sp}.jsonl")
            same_order = list(new) == list(ref)
            print(f"[ref] {sp}: {len(new)} rows vs {len(ref)}; same id set {set(new) == set(ref)}; same order "
                  f"{same_order}; targets identical {sum(new[c] == ref.get(c) for c in new)}/{len(new)}")

    def mk(cls, path, train, max_len, gt=True, limit=0):
        kw = {}
        if gt and train:
            kw["cue_gt"] = CueGT(a.cue_gt, a.cue_dropout, "sample", 0.95, 0.02, "oof")
            if a.unit_cue_gt:
                kw["unit_cue"] = UnitCueGT(a.unit_cue_gt, a.unit_cue_dropout, a.unit_cue_grades)
        return cls(path, tk, True, max_len, limit=limit, backbone=a.backbone, train=train, **kw)

    # 2. lengths (max_len huge -> nothing cut)
    BIG = 10 ** 6
    end_n = len(tk("<|im_end|>", add_special_tokens=False)["input_ids"])
    ds = mk(LenDS, data / "train.jsonl", True, BIG)
    tot_max = np.zeros(len(ds), int)
    for k in range(a.passes):
        pl, tl = np.zeros(len(ds), int), np.zeros(len(ds), int)
        for i in range(len(ds)):
            it = ds[i]
            pl[i] = len(it["prompt_ids"])
            tl[i] = len(it["input_ids"]) - pl[i]
            assert sum(x == ds.img_id for x in it["prompt_ids"]) == n_img
        tot = pl + tl
        tot_max = np.maximum(tot_max, tot)
        print(f"[len] train gt-cue pass {k + 1}: prompt {pct(pl)} | target(+end {end_n}) {pct(tl)} | total {pct(tot)}")
        for m in a.max_len:
            print(f"[len]   max-len {m}: targets cut {int((tot > m).sum())} / {len(ds)}")
    ds_o = mk(LenDS, data / "train.jsonl", True, BIG, gt=False)      # the jsonl prompt itself (stage 1, --cue-source oof)
    tot_o = np.array([len(ds_o[i]["input_ids"]) for i in range(len(ds_o))])
    tot_max = np.maximum(tot_max, tot_o)
    print(f"[len] train jsonl prompt (oof lines): total {pct(tot_o)}")
    for m in a.max_len:
        print(f"[len]   max-len {m}: targets cut {int((tot_o > m).sum())} / {len(ds_o)}; "
              f"over all passes (per-row max) {int((tot_max > m).sum())}")
    ds_v = mk(LenDS, data / "val.jsonl", True, BIG, gt=False)        # val loss rows: prompt as stored + target
    tot_v = np.array([len(ds_v[i]["input_ids"]) for i in range(len(ds_v))])
    print(f"[len] val ({len(ds_v)} rows, first 400 are the val-loss rows): total {pct(tot_v)}; first 400 {pct(tot_v[:400])}")
    for m in a.max_len:
        print(f"[len]   max-len {m}: val targets cut {int((tot_v > m).sum())}; in the first 400: {int((tot_v[:400] > m).sum())}")
    if a.val600:
        ds_i = mk(LenDS, a.val600, False, BIG)
        pl = np.array([len(ds_i[i]["prompt_ids"]) for i in range(len(ds_i))])
        print(f"[len] val600 inference prompts ({len(ds_i)} rows): {pct(pl)}")

    # 3. real batches
    ds_r = mk(p_arm.PDataset, data / "train.jsonl", True, a.use_max_len, limit=a.micro_bs * a.batches)
    print(f"[join] first {len(ds_r)} train rows without Pillar features: {ds_r.n_missing}")
    for name, path in (("train", data / "train.jsonl"), ("val", data / "val.jsonl"), ("val600", a.val600)):
        if not path:
            continue
        store = p_arm.PillarStore(a.pillar_root)
        ids = list(ids_targets(path))
        print(f"[join] {name}: {len(ids)} rows, without Pillar features {sum(not store.has(c) for c in ids)}")
    col = functools.partial(collate, pad_id=tk.pad_token_id, train=True)
    ok = 0
    for bi in range(a.batches):
        items = [ds_r[bi * a.micro_bs + j] for j in range(a.micro_bs)]
        for j, it in enumerate(items):
            lab = [x for x in it["labels"] if x != -100]
            tgt = ds_r.rows[bi * a.micro_bs + j]["target"]
            full = tk(tgt, add_special_tokens=False)["input_ids"] + ds_r.end_ids
            ok += lab == full
        b = col(items)
        print(f"[batch {bi}] input_ids {tuple(b['input_ids'].shape)} labels {tuple(b['labels'].shape)} "
              f"tok {tuple(b['tok'].shape)} {b['tok'].dtype} mask(units) {tuple(b['mask'].shape)} {b['mask'].dtype} "
              f"img tokens/row {(b['input_ids'] == ds_r.img_id).sum(1).tolist()} "
              f"finite {bool(torch.isfinite(b['tok'].float()).all() and torch.isfinite(b['mask'].float()).all())} "
              f"z-valid cells/row {b['tok'][..., -1].float().sum(1).int().tolist()} "
              f"units present/row {(b['mask'][..., p_arm.UNIT_DIM].float() > 0.5).sum(1).tolist()}")
    print(f"[batch] label part == full target + <|im_end|> in {ok}/{a.micro_bs * a.batches} rows")
    print("dry run done")


if __name__ == "__main__":
    main()
