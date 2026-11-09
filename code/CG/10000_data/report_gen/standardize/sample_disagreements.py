#!/usr/bin/env python3
"""Sample std14-vs-disease14 disagreements for manual review.

Takes <prefix>_rows.csv from check_std14.py; samples binary disagreements first (positive on one side only),
then 4-class ones of each kind, and writes the std record, the d14 quotes/why and the report next to each.
Usage: python sample_disagreements.py --prefix full/full --n 30 --out disagreements_review.txt
"""
import argparse
import json
import os

import pandas as pd

DATA = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data"
OUT = os.path.join(DATA, "report_gen", "standardize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.prefix + "_rows.csv")
    d = d[d.d14_status.notna() & (d.d14_status != d.std_status)].copy()
    d["kind"] = d.d14_status + "->" + d.std_status
    binary = d[(d.d14_status == "positive") ^ (d.std_status == "positive")]
    nb = min(len(binary), a.n * 2 // 3)
    pick = [binary.sample(nb, random_state=0)] if nb else []
    rest = d.drop(pick[0].index) if nb else d
    kinds = rest.kind.value_counts()
    left = a.n - nb
    for k in kinds.index:  # spread the remaining picks over kinds, most frequent first
        if left <= 0:
            break
        m = max(1, min(left, round((a.n - nb) * kinds[k] / kinds.sum())))
        pick.append(rest[rest.kind == k].sample(min(m, kinds[k]), random_state=0))
        left -= m
    s = pd.concat(pick)
    lab = {}
    for l in open(os.path.join(DATA, "disease14_qwen.jsonl")):
        j = json.loads(l)
        lab[j["study_id"]] = j["answer"]
    rep = json.load(open(os.path.join(OUT, "reports_std_in.json")))
    with open(a.out, "w") as f:
        f.write(f"{len(d)} 4-class disagreements; kinds:\n{d.kind.value_counts().to_string()}\n"
                f"binary disagreements: {len(binary)}\n")
        for i, (_, r) in enumerate(s.iterrows(), 1):
            q = lab[r.exam_id][r.disease]
            f.write(f"\n{'=' * 90}\n[{i}] {r.exam_id} {r['loc']} {r.disease}: d14={q['category']} std={r.std_status}\n"
                    f"  d14 quotes: {q['quotes']} | why: {q['why']}\n  std source: {r.source}\n"
                    f"  std normalized: {r.normalized}\n{'-' * 30} report\n{rep[r.exam_id]['report'].strip()[:3000]}\n")
    s[["exam_id", "loc", "disease", "d14_cat", "std_status", "kind", "source", "normalized"]].to_csv(
        a.out.replace(".txt", ".csv"), index=False)
    print(len(s), "sampled")


if __name__ == "__main__":
    main()
