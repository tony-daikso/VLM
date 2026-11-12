#!/usr/bin/env python3
"""Agent L (11-12): inference-only "top-1 vertebra fallback" for the unit cue line (arm T_unit_gt_pil, val600 only).

What the LLM sees: the unit line lists the raw 2-decimal score of every scored level T8-L5 (no marker, no cut). In
training (train.py --unit-cue-gt) a labelled-positive vertebra always showed a value drawn from the train OOF scores of
label-positive vertebrae at / above the unit threshold (pools["vertebra"]["pos"]), a labelled-negative one a value
below it, so "a level at / above the threshold" is what the model learned to copy into the report.

Fallback (prompt data rewrite only, no retraining): when the case-level fracture cue of the row's classifier line is
>= the method-1 threshold (cue_gt.json) and no vertebra score is >= the unit threshold, the top-1 vertebra (highest
score, first one on ties) is shown with FILL = the median of pools["vertebra"]["pos"] (train OOF only, 2 decimals);
every other number and line is unchanged ("unit_scores" is kept equal to the line).
  fb_all  : every such row.
  fb_gate : only when the original top-1 score >= GATE. GATE comes from TRAIN OOF only (stored train prompts = OOF
            classifier cue, OOF unit scores): among train rows with fracture cue >= threshold and no vertebra >=
            the unit threshold, the top-1 cut-off with the largest Youden J (TPR - FPR) for "case has a labelled
            fractured vertebra in T8-L5" (labels_train vert, any 1) vs not; ties -> the lowest cut-off.
Rules are fixed here before any val result is read; val rows are only rewritten, never scored. Test rows are refused.
No ids / text are printed; --stats-out holds aggregates only.
usage (A6000): python level_fallback.py --train improve/data_unit_pil/train.jsonl \
    --unit-cue improve/data_unit_pil/unit_cue_pil.json --cue-gt cue/cue_gt.json \
    --val600 improve/data/val600_unit_pil.jsonl --out-prefix improve/data/val600_unit_pil --stats-out <json>
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import UNIT_PREFIX, parse_cue, render_unit_line, replace_unit_line  # noqa: E402

EPS = 1e-9


def rows_of(p):
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                raise SystemExit("test split is locked")
            yield r


def candidate(r, frac_thr, unit_thr):
    """-> (top-1 index, top-1 score) when the row is a fallback candidate (fracture cue positive, vertebra scores
    present, none at / above the unit threshold), else None."""
    cue, _ = parse_cue(r["prompt"])
    if cue.get("fracture") is None or cue["fracture"] < frac_thr - EPS:
        return None
    vert = r["unit_scores"]["vert"]
    sc = [(v, -i) for i, v in enumerate(vert) if v is not None]
    if not sc or not any(ln.startswith(UNIT_PREFIX) and "vertebral fracture " in ln for ln in r["prompt"].split("\n")):
        return None
    v, ni = max(sc)
    if v >= unit_thr - EPS:
        return None
    return -ni, v


def pick_gate(pairs):
    """pairs: [(top-1 score, label 1/0)] -> (gate, table) by max Youden J, ties -> lowest cut-off."""
    P = sum(y for _, y in pairs)
    N = len(pairs) - P
    best, tab = None, []
    for g in sorted({s for s, _ in pairs}):
        tp = sum(y for s, y in pairs if s >= g - EPS)
        fp = sum(1 - y for s, y in pairs if s >= g - EPS)
        j = tp / max(P, 1) - fp / max(N, 1)
        f1 = 2 * tp / max(tp + fp + P, 1)
        tab.append({"gate": g, "tp": tp, "fp": fp, "tpr": tp / max(P, 1), "fpr": fp / max(N, 1), "youden": j, "f1": f1})
        if best is None or j > best[0] + EPS:
            best = (j, g)
    return best[1], tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--unit-cue", required=True)
    ap.add_argument("--cue-gt", required=True)
    ap.add_argument("--val600", required=True)
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_fb_all.jsonl and <prefix>_fb_gate.jsonl")
    ap.add_argument("--stats-out", required=True)
    a = ap.parse_args()
    z = json.load(open(a.unit_cue))
    unit_thr = z["thresholds"]["vertebra"]
    frac_thr = json.load(open(a.cue_gt))["thresholds"]["fracture"]
    pool = z["pools"]["vertebra"]["pos"]
    fill = round(statistics.median(pool), 2)
    assert fill >= unit_thr
    # ---- gate: train OOF only
    pairs, n_train = [], 0
    for r in rows_of(a.train):
        assert r["split"] == "train"
        n_train += 1
        c = candidate(r, frac_thr, unit_thr)
        if c is None:
            continue
        lab = z["labels_train"][r["case_id"]]["vert"]
        pairs.append((c[1], int(any(y == 1 for y in lab))))
    gate, tab = pick_gate(pairs)
    f1_gate = max(tab, key=lambda t: (t["f1"], -t["gate"]))["gate"]
    st = {"unit_thr": unit_thr, "fracture_cue_thr": frac_thr, "fill": fill, "pos_pool_n": len(pool),
          "pos_pool_quartiles": [round(x, 3) for x in statistics.quantiles(pool, n=4)],
          "train": {"rows": n_train, "candidates": len(pairs), "vertebral_label_pos": sum(y for _, y in pairs),
                    "gate_youden": gate, "gate_f1_for_reference": f1_gate, "criterion": "max Youden J, ties -> lowest",
                    "table": tab}}
    # ---- val600 rewrite (no scoring)
    val = list(rows_of(a.val600))
    assert all(r["split"] == "val" for r in val)
    for tag, g in (("fb_all", None), ("fb_gate", gate)):
        n_cand = n_fb = 0
        out = Path(f"{a.out_prefix}_{tag}.jsonl")
        assert out.resolve() != Path(a.val600).resolve()
        with open(out, "w") as fo:
            for r in val:
                r = json.loads(json.dumps(r))
                c = candidate(r, frac_thr, unit_thr)
                if c is not None:
                    n_cand += 1
                    if g is None or c[1] >= g - EPS:
                        n_fb += 1
                        us = r["unit_scores"]
                        us["vert"][c[0]] = fill
                        old = r["prompt"]
                        r["prompt"] = replace_unit_line(old, render_unit_line(us["kid"], us["vert"]))
                        a_, b_ = old.split("\n"), r["prompt"].split("\n")
                        assert len(a_) == len(b_) and sum(x != y for x, y in zip(a_, b_)) == 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        st[tag] = {"rows": len(val), "candidates": n_cand, "fallback_applied": n_fb, "file": out.name}
    Path(a.stats_out).write_text(json.dumps(st, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "train"}))
    print("train:", {k: v for k, v in st["train"].items() if k != "table"})


if __name__ == "__main__":
    main()
