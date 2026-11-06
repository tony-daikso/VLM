"""Build labels csv + split csv from the Qwen disease14 jsonl, and optionally pick a pilot subset.

Binarisation: yes / probably_yes -> 1 ; no / probably_no / not_mentioned -> 0 ; unclear -> NaN (excluded).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--split-out", required=True)
    ap.add_argument("--pilot-out", default=None)
    ap.add_argument("--pilot-n", type=int, default=60)
    ap.add_argument("--pilot-enrich", nargs="*", default=["hepatic_steatosis", "surgically_absent_gallbladder"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows, splits = [], []
    for line in open(a.jsonl):
        r = json.loads(line)
        d = {"study_id": r["study_id"]}
        for k, v in r["answer"].items():
            c = v["category"]
            d[k] = 1.0 if c in POS else 0.0 if c in NEG else np.nan
        rows.append(d)
        splits.append({"study_id": r["study_id"], "split": r["split"]})
    lab = pd.DataFrame(rows).drop_duplicates("study_id")
    spl = pd.DataFrame(splits).drop_duplicates("study_id")
    lab.to_csv(a.labels_out, index=False)
    spl.to_csv(a.split_out, index=False)
    print(len(lab), "studies;", spl.split.value_counts().to_dict())
    print("positives:", lab.drop(columns="study_id").sum().astype(int).to_dict())

    if a.pilot_out:
        rng = np.random.default_rng(a.seed)
        df = lab.merge(spl, on="study_id")
        per_split = {"train": a.pilot_n // 2, "val": a.pilot_n // 4, "test": a.pilot_n - a.pilot_n // 2 - a.pilot_n // 4}
        chosen = []
        for s, n in per_split.items():
            sub = df[df.split == s]
            picked = set()
            for f in a.pilot_enrich:          # ~1/3 of each split positive for each control finding
                pos = sub[sub[f] == 1].study_id.values
                picked |= set(rng.choice(pos, size=min(len(pos), n // 3), replace=False))
            rest = sub[~sub.study_id.isin(picked)].study_id.values
            picked |= set(rng.choice(rest, size=max(0, n - len(picked)), replace=False))
            chosen += sorted(picked)[:n]
        with open(a.pilot_out, "w") as f:
            f.write("\n".join(chosen) + "\n")
        print("pilot", len(chosen), "cases ->", a.pilot_out)


if __name__ == "__main__":
    main()
