#!/usr/bin/env python3
"""Method 1: does the Qwen re-extraction of the draft agree with what the template meant to say?
usage: python m1_consistency.py <eval results dir with per_case_categories.jsonl> <A|B>
per_case_categories rows: {case_id, <disease>: [reference category, generated category]}
expected generated category: pos/verify -> yes|probably_yes ; neg (non-silent) -> no|probably_no ;
silent negative / omit -> not positive (not_mentioned expected).
Writes <results dir>/m1_consistency.csv"""
import json
import sys

import pandas as pd

from m1_common import D14, OUT, ZH

res, var = sys.argv[1], sys.argv[2]
silent = json.load(open(OUT / "sentence_bank.json"))["silent_negative"]
dec = pd.read_csv(OUT / f"val_decisions_{var}.csv", index_col=0)
rows = []
for line in open(f"{res}/per_case_categories.jsonl"):
    r = json.loads(line)
    for d in D14:
        if d not in r or r[d] is None:
            continue
        g = r[d][1]; x = dec.loc[r["case_id"], d]
        exp = "pos" if x in ("pos", "verify") else ("neg" if (x == "neg" and not silent[d]) else "silent")
        ok = (g in ("yes", "probably_yes")) if exp == "pos" else (g in ("no", "probably_no")) if exp == "neg" else (g not in ("yes", "probably_yes"))
        rows.append(dict(case_id=r["case_id"], disease=d, 疾病=ZH[d], decision=x, expected=exp, gen=g, ok=ok))
R = pd.DataFrame(rows)
R.to_csv(f"{res}/m1_consistency.csv", index=False)
print(f"overall agreement {R.ok.mean():.3f} ({R.ok.sum()}/{len(R)})")
print(R.groupby(["疾病", "expected"]).ok.agg(["mean", "count"]).round(3).to_string())
print(R[~R.ok].groupby(["疾病", "decision", "gen"]).size().to_string())
