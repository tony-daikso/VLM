#!/usr/bin/env python3
"""Improvement experiments (agent F, 2026-11-11): prompt variants of the method-4 jsonl files. No report text is
printed; outputs (which hold the targets) stay on A6000 under m4/improve/.

  thresholds  : sensitivity-biased cue thresholds from TRAIN OOF only
                  s90 = largest t with train OOF sensitivity >= 0.90 (owl binarisation, unclear left out)
                  s80 / s95 = the same for 0.80 / 0.95; m1a = method 1A positive threshold (method1/thresholds.csv t_neg, sens >= .95)
  remap       : rewrite the classifier line of val rows so that a raw val score >= t_new is shown as a value >= t_f1
                (the side of the threshold the gt-dropout models learned) and a score < t_new as a value < t_f1;
                monotone piecewise-linear map of the RAW score (oof/out/scores_trval_prob.csv), then 2 decimals.
                Only val rows; train rows are refused.
  oracle      : DIAGNOSTIC upper bound only (coordinator request 11-11): the classifier line of val rows rebuilt from the
                report labels (labels_disease14, owl binarisation), same mapping as train.py gt_dropout "sample" (value
                drawn from the train OOF pool on the label's side of t_f1) but no dropout; unclear -> left out.
                Never a model result; val only.
  nocue       : drop the classifier line (the existing "no score" format) from every row (train / val / subset)
usage:
  python make_variants.py thresholds --scores <scores_trval_prob.csv> --cue-gt <cue_gt.json> --m1 <thresholds.csv> --out thr.json
  python make_variants.py remap --in val_sub600.jsonl --out val600_s90.jsonl --thr thr.json --which s90 --scores ...
  python make_variants.py nocue --in train.jsonl --out data_nocue/train.jsonl
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import DISEASES, gt_cue_probs, owl_binary, parse_cue, replace_cue  # noqa: E402


def sens_threshold(pos_scores, target):
    """largest t with mean(pos >= t) >= target"""
    s = sorted(pos_scores)
    k = int(math.floor((1 - target) * len(s) + 1e-9))
    return s[k]


def cmd_thresholds(a):
    z = json.load(open(a.cue_gt))
    lab, t_f1 = z["labels_train"], z["thresholds"]
    m1 = {r["disease"]: r for r in csv.DictReader(open(a.m1))}
    pos = {d: [] for d in DISEASES}
    neg = {d: [] for d in DISEASES}
    for r in csv.DictReader(open(a.scores)):
        if r["split"] != "train":
            continue
        y = lab[r["study_id"]]
        for k, d in enumerate(DISEASES):
            if y[k] is None or r[d] == "":
                continue
            (pos if y[k] == 1 else neg)[d].append(float(r[d]))
    out = {"t_f1": t_f1, "s80": {}, "s90": {}, "s95": {}, "m1a": {}, "train_ops": {}}
    for d in DISEASES:
        out["s80"][d] = sens_threshold(pos[d], 0.80)
        out["s90"][d] = sens_threshold(pos[d], 0.90)
        out["s95"][d] = sens_threshold(pos[d], 0.95)
        out["m1a"][d] = float(m1[d]["t_neg"])
        ops = {}
        for name in ("t_f1", "s80", "s90", "s95", "m1a"):
            t = out[name][d]
            ops[name] = {"t": t, "sens": sum(v >= t for v in pos[d]) / len(pos[d]),
                         "spec": sum(v < t for v in neg[d]) / len(neg[d])}
        out["train_ops"][d] = ops
    out["meta"] = {"n_pos": {d: len(pos[d]) for d in DISEASES}, "n_neg": {d: len(neg[d]) for d in DISEASES},
                   "source": "train OOF only (scores_trval_prob.csv split=train, cue_gt.json labels_train)"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{'disease':28s} {'t_f1':>6s} {'s80':>7s} {'s90':>7s} {'m1a':>7s} | train sens/spec at t_f1, s80, s90, m1a")
    for d in DISEASES:
        o = out["train_ops"][d]
        print(f"{d:28s} {out['t_f1'][d]:6.3f} {out['s80'][d]:7.4f} {out['s90'][d]:7.4f} {out['m1a'][d]:7.4f} | "
              + "  ".join(f"{o[n]['sens']:.3f}/{o[n]['spec']:.3f}" for n in ("t_f1", "s80", "s90", "m1a")))


def remap_value(p, t_new, t_ref):
    """raw score -> 2-decimal value on the same side of t_ref as p is of t_new (monotone, piecewise linear)."""
    if t_new >= t_ref:                      # not a sensitivity shift: keep the value
        return round(p, 2)
    if p >= t_new:
        q = t_ref + (p - t_new) * (1 - t_ref) / (1 - t_new)
        v = math.ceil(q * 100 - 1e-9) / 100
        v = max(v, math.ceil(t_ref * 100 - 1e-9) / 100)
        return min(v, 1.0)
    q = p * t_ref / t_new
    v = math.floor(q * 100 + 1e-9) / 100
    while v >= t_ref:                       # rendered value must stay below t_ref
        v -= 0.01
    return max(round(v, 2), 0.0)


def cmd_remap(a):
    thr = json.load(open(a.thr))
    t_new, t_ref = thr[a.which], thr["t_f1"]
    raw = {r["study_id"]: r for r in csv.DictReader(open(a.scores)) if r["split"] == "val"}
    n = n_shift = n_mismatch = 0
    cnt = {d: [0, 0, 0] for d in DISEASES}     # [positive before (t_f1), positive after, scored]
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            if r["split"] != "val":
                sys.exit("remap is for val rows only")
            cue, _ = parse_cue(r["prompt"])
            new = {}
            for d, v in cue.items():
                p = float(raw[r["case_id"]][d])
                if abs(round(p, 2) - v) > 0.0051:
                    n_mismatch += 1
                new[d] = remap_value(p, t_new[d], t_ref[d])
                cnt[d][0] += v >= t_ref[d]
                cnt[d][1] += new[d] >= t_ref[d]
                cnt[d][2] += 1
                n_shift += (v >= t_ref[d]) != (new[d] >= t_ref[d])
            if cue:
                r["prompt"] = replace_cue(r["prompt"], new)
            r["cue_variant"] = a.which
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows; {n_shift} cells moved to the positive side; raw/prompt mismatches {n_mismatch}")
    for d in DISEASES:
        print(f"  {d:28s} positive cues {cnt[d][0]:4d} -> {cnt[d][1]:4d} / {cnt[d][2]}")


def cmd_oracle(a):
    import random
    pools = json.load(open(a.cue_gt))["pools"]
    lab = {}
    for line in open(a.labels):
        r = json.loads(line)
        if r.get("split") != "val":
            continue                                     # test rows skipped unread
        ans = r["answer"] if isinstance(r["answer"], dict) else json.loads(r["answer"])
        lab[str(r["study_id"])] = [owl_binary(ans[d]["category"]) for d in DISEASES]
    n = k = 0
    npos = 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            if r["split"] != "val":
                sys.exit("oracle cues are for val rows only")
            probs, _, _ = gt_cue_probs(lab[r["case_id"]], pools, random.Random(f"oracle:{r['case_id']}"), 0.0, "sample")
            r["prompt"] = replace_cue(r["prompt"], probs)
            r["cue_variant"] = "oracle"
            npos += sum(1 for v in lab[r["case_id"]] if v == 1)
            k += len(probs)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows, {k} cue cells from labels ({npos} positive)")


def cmd_nocue(a):
    n = k = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            cue, _ = parse_cue(r["prompt"])
            if cue:
                r["prompt"] = replace_cue(r["prompt"], {})
                k += 1
            assert parse_cue(r["prompt"])[1] is None
            r["cue_variant"] = "nocue"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows, classifier line removed from {k}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thresholds")
    t.add_argument("--scores", required=True)
    t.add_argument("--cue-gt", required=True)
    t.add_argument("--m1", required=True)
    t.add_argument("--out", required=True)
    r = sub.add_parser("remap")
    r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--thr", required=True)
    r.add_argument("--which", required=True, choices=["s80", "s90", "s95", "m1a"])
    r.add_argument("--scores", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--in", dest="inp", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--cue-gt", required=True)
    o.add_argument("--labels", required=True)
    n = sub.add_parser("nocue")
    n.add_argument("--in", dest="inp", required=True)
    n.add_argument("--out", required=True)
    a = ap.parse_args()
    {"thresholds": cmd_thresholds, "remap": cmd_remap, "nocue": cmd_nocue, "oracle": cmd_oracle}[a.cmd](a)


if __name__ == "__main__":
    main()
