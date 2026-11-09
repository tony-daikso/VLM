#!/usr/bin/env python3
"""Sanity-check input for evaluate.py: the reference report itself as the 'generated' text (val only).

Picks --per-class val cases positive (yes/probably_yes) for each of the 14 classes plus --n-random random val cases,
seed fixed, and writes {case_id, text} rows (report text: DUA, keep the output off git / outside the repo).
usage: python sanity_make_inputs.py --out <scratch>/sanity_gen.jsonl --ids-out $MERLIN_DATA/report_gen/eval/sanity_ids.txt
(both outputs hold DUA text / study ids: keep them under data/ or a scratch dir, never in the code folder)
"""
import argparse
import json
import os
import random
from pathlib import Path

D14 = Path(os.environ.get("MERLIN_DATA", "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin")) / "disease14"
POS = {"yes", "probably_yes"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-class", type=int, default=2)
ap.add_argument("--n-random", type=int, default=8)
ap.add_argument("--seed", type=int, default=20261109)
ap.add_argument("--ids-out", default="")
a = ap.parse_args()
rng = random.Random(a.seed)
lab = [json.loads(l) for l in open(D14 / "labels_disease14.jsonl")]
val = [r for r in lab if r["split"] == "val"]
chosen = []
for d in val[0]["answer"]:
    pool = sorted(r["study_id"] for r in val if r["answer"][d]["category"] in POS and r["study_id"] not in chosen)
    chosen += rng.sample(pool, min(a.per_class, len(pool)))
rest = sorted(r["study_id"] for r in val if r["study_id"] not in chosen)
chosen += rng.sample(rest, a.n_random)
rep = json.load(open(D14 / "reports.json"))
with open(a.out, "w") as f:
    for c in chosen:
        assert rep[c]["split"] == "val"
        f.write(json.dumps({"case_id": c, "text": rep[c]["report"]}) + "\n")
if a.ids_out:
    Path(a.ids_out).write_text("\n".join(chosen) + "\n")
print(len(chosen), "val cases ->", a.out)
