#!/usr/bin/env python3
"""Cue file for --cue-source gt_dropout (train.py) and the error-cue test (infer.py --cue-flip-*).

Writes <out>/cue_gt.json:
  labels_train  {case_id: [14 x 1/0/null]}  report labels of the TRAIN split only (owl binarisation:
                yes/probably_yes -> 1; no/probably_no/not_mentioned -> 0; unclear -> null). Val/test labels are never
                written, so a val prompt cannot be built from labels by accident.
  thresholds    {disease: t}  method-1 decision threshold t_f1 (fit on train OOF; method1/thresholds.csv)
  pools         {disease: {"pos": [...], "neg": [...]}}  train OOF probabilities (rounded to 2 decimals, as printed
                in the prompt) of label-positive cases with value >= t and of label-negative cases with value < t.
                gt_dropout "sample" draws the cue of a positive / negative finding from these; the error-cue test
                draws the flipped value from the opposite pool. Train OOF only -> no val information.
  meta          input md5s, counts.
No report text. The file holds case ids + labels: keep it in the data folder / A6000 m4/, never in git.
usage: python build_cue_gt.py --out <dir>   (defaults point at the Mac data folder; MERLIN_DATA overrides)
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m4_common import DISEASES, owl_binary  # noqa: E402

MERLIN = Path(os.environ.get("MERLIN_DATA", "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin"))


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(MERLIN / "disease14/labels_disease14.jsonl"))
    ap.add_argument("--scores", default=str(MERLIN / "report_gen/oof/out/scores_trval_prob.csv"))
    ap.add_argument("--thresholds", default=str(MERLIN / "report_gen/method1/thresholds.csv"))
    ap.add_argument("--threshold-col", default="t_f1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    thr = {r["disease"]: float(r[a.threshold_col]) for r in csv.DictReader(open(a.thresholds))}
    assert set(thr) == set(DISEASES), "threshold table does not cover the 14 classes"
    labels = {}
    for line in open(a.labels):
        r = json.loads(line)
        if r.get("split") != "train":
            continue                                             # val / test labels never enter the cue file
        ans = r["answer"] if isinstance(r["answer"], dict) else json.loads(r["answer"])
        labels[str(r["study_id"])] = [owl_binary(ans[d]["category"]) for d in DISEASES]
    pools = {d: {"pos": [], "neg": []} for d in DISEASES}
    n_score = 0
    for r in csv.DictReader(open(a.scores)):
        if r["split"] != "train":
            continue
        cid = r["study_id"]
        if cid not in labels:
            sys.exit(f"{cid}: train score without a train label")
        n_score += 1
        for k, d in enumerate(DISEASES):
            y, s = labels[cid][k], r.get(d, "")
            if y is None or s == "":
                continue
            v = round(float(s), 2)
            if y == 1 and v >= thr[d]:
                pools[d]["pos"].append(v)
            elif y == 0 and v < thr[d]:
                pools[d]["neg"].append(v)
    for d in DISEASES:
        for s in ("pos", "neg"):
            pools[d][s].sort()
            assert pools[d][s], f"empty {s} pool for {d}"
    n_lab = {d: {"pos": sum(1 for y in labels.values() if y[k] == 1), "neg": sum(1 for y in labels.values() if y[k] == 0),
                 "unclear": sum(1 for y in labels.values() if y[k] is None)} for k, d in enumerate(DISEASES)}
    meta = {"labels_md5": md5(a.labels), "scores_md5": md5(a.scores), "thresholds_md5": md5(a.thresholds),
            "threshold_col": a.threshold_col, "n_train_labels": len(labels), "n_train_scores": n_score,
            "label_counts": n_lab, "pool_sizes": {d: {s: len(pools[d][s]) for s in ("pos", "neg")} for d in DISEASES}}
    os.makedirs(a.out, exist_ok=True)
    Path(a.out, "cue_gt.json").write_text(json.dumps({"labels_train": labels, "thresholds": thr, "pools": pools,
                                                      "meta": meta}))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
