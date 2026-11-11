#!/usr/bin/env python3
"""Agreement between "merge" scoring and a real Qwen re-extraction of the fill14 text (counts only, no text).
For every group: per_case_categories.jsonl ([reference, generated]) of <root>/<g>_<split>_fill (real) and
<root>/<g>_<split>_fill_merge (merge); cells split into filled (fill14 wrote the template) and untouched.
Reports 6-way category agreement and presence agreement (yes/probably_yes vs the rest), plus the real-extraction
category distribution of filled cells (did Qwen read the template as intended?).
usage: fill14_agree.py --root m4/eval/results_fill14 --split val600 --gen-root m4/fill14/gen U_oof U_gt T_oof T_gt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

POS = {"yes", "probably_yes"}


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l) for l in open(p) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val600")
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    out = {}
    for g in a.groups:
        real = load(Path(a.root) / f"{g}_{a.split}_fill" / "per_case_categories.jsonl")
        merge = load(Path(a.root) / f"{g}_{a.split}_fill_merge" / "per_case_categories.jsonl")
        filled = {r["case_id"]: r["filled"] for r in map(json.loads, open(Path(a.gen_root) / g / f"{a.split}.jsonl"))}
        ids = sorted(set(real) & set(merge))
        res = {"n_cases": len(ids)}
        for part in ("filled", "untouched", "all"):
            n = same6 = samep = 0
            dist = Counter()
            for c in ids:
                for d, v in real[c].items():
                    if d == "case_id":
                        continue
                    rcat, mc = v[1], merge[c][d][1]
                    isf = d in filled.get(c, {})
                    if (part == "filled" and not isf) or (part == "untouched" and isf):
                        continue
                    n += 1
                    same6 += rcat == mc
                    samep += (rcat in POS) == (mc in POS)
                    if part == "filled":
                        dist[f"{filled[c][d]}->{rcat}"] += 1
            res[part] = {"n_cells": n, "agree_6way": same6 / n if n else None,
                         "agree_presence": samep / n if n else None}
            if part == "filled":
                res[part]["real_category_of_filled"] = dict(dist.most_common())
        # cases whose 14 presence calls all agree
        res["cases_all_presence_agree"] = sum(
            all((real[c][d][1] in POS) == (merge[c][d][1] in POS) for d in real[c] if d != "case_id") for c in ids)
        out[g] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
