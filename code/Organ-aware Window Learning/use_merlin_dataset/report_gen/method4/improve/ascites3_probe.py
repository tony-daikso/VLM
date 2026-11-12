#!/usr/bin/env python3
"""Agent N (11-12): ascites "large" probe (large vs trace / small / moderate) for a three-level ascites amount cue in the
method-4 grade line. CPU only.

Why: on val600 every ascites-amount error of A+ was off by one grade, mostly "large" written as "moderate"; the
existing asc_amount score (improve/probe_pillar, P(moderate or large)) is binary.

labels  : m4/inputs/targets_v2, diseases.ascites: status == pos and attributes.amount in trace / small / moderate / large
          (structured fields only; no report text is read or printed; targets_test is never opened).
          y_large = 1 (large) / 0 (trace, small, moderate) / nan. The binary moderate+ label rebuilt from the same field
          must equal the peer's attrs.csv asc_amount (assert), so the labelled set is the peer's.
method  : exactly the peer's asc_amount recipe (pillar/optionA/build_aplus.py grade_scores, imported read-only):
          Pillar organ-unit probe (optA.PF.unit_X ascites units, nested-CV standardised LR) stacked with
          logit(OWL ascites probability) by optA.stack_lr, sigmoid of the decision function.
folds   : the official OWL ascites stack folds (pillar_optA/in/cases.csv fold_ascites). TRAIN = out-of-fold (labelled
          and unlabelled train rows are scored by the model of their own fold), VAL = full-train refit.
          Threshold (F1) and pos / neg pools: train OOF only. Nothing is tuned on val.
report  : val AUROC (large vs not large, labelled val cases) with stratified bootstrap 95% CIs (2,000) for the probe,
          the existing asc_amount score and the OWL ascites probability; paired difference probe - asc_amount.
          decision.add_to_prompt = the paired difference CI lower bound > 0 (rule fixed before val was read).
          Three-level accuracy (thresholds from train OOF) is reported only.
writes  : <out>/probe_scores_ascites3.csv (case_id, split, asc_large), unit_cue_ascites3.json (= --cue-json + key
          asc_large: labels_train, threshold, pools), check.json (aggregates only), fit.log. Files hold case ids: they
          stay on A6000, never in git.
usage (A6000, radar env): python ascites3_probe.py --out improve/probe_ascites3 [--threads 4]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
H = Path(os.path.expanduser("~/Desktop/Tony/VLM/OWL"))
M4 = H / "report_gen/m4"
OPTA = H / "pillar_optA"
KEY = "asc_large"
AMT = {"trace": 0, "small": 1, "moderate": 2, "large": 3}
B = 2000


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def read_amount(ids, split):
    """-> amount level [N] (nan / 0..3) of the report-positive ascites cases with the amount written."""
    idx = {c: i for i, c in enumerate(ids)}
    g = np.full(len(ids), np.nan)
    pos = np.zeros(len(ids), bool)
    for sp in ("train", "val"):                                 # the test targets are never opened
        for line in open(M4 / f"inputs/targets_v2/targets_{sp}.jsonl"):
            r = json.loads(line)
            assert r["split"] == sp
            i = idx.get(r["case_id"])
            if i is None:
                continue
            assert split[i] == sp
            x = r["diseases"]["ascites"]
            if x["status"] != "pos":
                continue
            pos[i] = True
            g[i] = AMT.get(x["attributes"].get("amount"), np.nan)
    return pos, g


def boot_auc(y, S, pairs, seed=11, b=B):
    """stratified case bootstrap. S {name: score}; pairs [(a, b)] -> paired a - b."""
    from scipy.stats import rankdata

    def auc(yy, ss):
        n1 = yy.sum()
        r = rankdata(ss)
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    A = {k: [] for k in S}
    for _ in range(b):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k in S:
            A[k].append(auc(y[i], S[k][i]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {k: {"auroc": auc(y, S[k]), "ci": ci(A[k])} for k in S}
    for a_, b_ in pairs:
        d = np.array(A[a_]) - np.array(A[b_])
        out[f"{a_}-{b_}"] = {"delta": out[a_]["auroc"] - out[b_]["auroc"], "ci": ci(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cue-json", default=str(M4 / "improve/probe_pillar/unit_cue_pillar.json"),
                    help="unit cue json to copy and extend (read-only)")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    sys.path.insert(0, str(OPTA / "code"))
    import optA as oa                                           # peer code, read-only (nested LR, stack, Pillar shards)
    import pandas as pd
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)
    except ImportError:
        pass

    C = pd.read_csv(OPTA / "in/cases.csv", usecols=["study_id", "split", "fold_ascites", "owl_ascites"])
    A = pd.read_csv(OPTA / "in/attrs.csv", usecols=["study_id", "asc_amount"]).set_index("study_id").reindex(C.study_id)
    ids, split = C.study_id.values.astype(str), C.split.values.astype(str)
    assert set(split) <= {"train", "val"}
    tr, va = split == "train", split == "val"
    pos, g = read_amount(ids, split)
    y2_peer = A["asc_amount"].values.astype(float)
    y2 = np.where(np.isnan(g), np.nan, (g >= 2).astype(float))
    assert (np.isnan(y2) == np.isnan(y2_peer)).all() and np.nansum(np.abs(y2 - y2_peer)) == 0, \
        "amount labels differ from the peer's asc_amount"
    y = np.where(np.isnan(g), np.nan, (g == 3).astype(float))
    counts = {sp: {"cases": int(m.sum()), "ascites_positive": int((m & pos).sum()),
                   "amount_written": int((m & ~np.isnan(g)).sum()),
                   "amount": {k: int((m & (g == v)).sum()) for k, v in AMT.items()}}
              for sp, m in (("train", tr), ("val", va))}
    log(f"[{time.strftime('%m-%d %H:%M')}] ascites3: {json.dumps(counts)}")

    pf = oa.PF()
    r, have = pf.rows(ids)
    assert have.all()
    fold = C["fold_ascites"].values
    ow = oa.logit(C["owl_ascites"].values)
    ytr = np.where(tr, y, np.nan)
    t0 = time.time()
    pu, info = oa.nested(pf.unit_X(r, "ascites"), ytr, fold, tr, va, oa.CS, "lr")
    log(f"  Pillar unit probe: C {info['C_outer']} / {info['C_refit']} ({time.time() - t0:.0f}s)")
    s = sig(oa.stack_lr(np.c_[ow, pu], ytr, fold, tr, va))
    assert not np.isnan(s).any()
    # the existing binary score (peer file, read-only), same case order asserted
    # the existing binary score (peer file, read-only). The probe is fitted on every cases.csv row like the peer's grade
    # scores; everything below (thresholds, pools, metrics, outputs) uses the peer's score-table cases in its order.
    pr = list(csv.DictReader(open(M4 / "improve/probe_pillar/probe_scores_pillar.csv")))
    ci = {c: i for i, c in enumerate(ids)}
    gix = np.array([ci[x["case_id"]] for x in pr])
    assert (split[gix] == np.array([x["split"] for x in pr])).all()
    n_fit = {"cases_csv_rows": int(len(ids)), "score_table_rows": int(len(pr)),
             "labelled_train_fit": int((tr & ~np.isnan(y)).sum())}
    ids, split, tr, va, pos, g, y, s = ids[gix], split[gix], tr[gix], va[gix], pos[gix], g[gix], y[gix], s[gix]
    s_amt = np.array([float(x["asc_amount"]) for x in pr])
    owl = C["owl_ascites"].values.astype(float)[gix]
    counts = {sp: {"cases": int(m.sum()), "ascites_positive": int((m & pos).sum()),
                   "amount_written": int((m & ~np.isnan(g)).sum()),
                   "amount": {k: int((m & (g == v)).sum()) for k, v in AMT.items()}}
              for sp, m in (("train", tr), ("val", va))}
    log(f"  score-table cases: {json.dumps(n_fit)} {json.dumps(counts)}")

    # thresholds / pools: train OOF only
    lt, lv = tr & ~np.isnan(y), va & ~np.isnan(y)
    t_f1 = oa.f1_thr(y[lt], s[lt])
    pools = {"pos": sorted(round(float(v), 2) for v in s[lt][(y[lt] == 1) & (s[lt] >= t_f1)]),
             "neg": sorted(round(float(v), 2) for v in s[lt][(y[lt] == 0) & (s[lt] < t_f1)])}
    t_amt = json.load(open(a.cue_json))["thresholds"]["asc_amount"]
    # for reference: a "large" threshold on the existing score, also from train OOF F1
    t_amt_large = oa.f1_thr(y[lt], s_amt[lt])
    log(f"  thresholds (train OOF F1): asc_large {t_f1}; existing asc_amount as a large score {t_amt_large}; "
        f"pools pos {len(pools['pos'])} neg {len(pools['neg'])}")

    res = {}
    for sp, m in (("train_oof", lt), ("val", lv)):
        S = {"probe": s[m], "asc_amount": s_amt[m], "owl": owl[m]}
        res[sp] = boot_auc(y[m], S, [("probe", "asc_amount"), ("probe", "owl")])
        res[sp]["_n"] = {"cases": int(m.sum()), "large": int(y[m].sum())}
        gg = g[m].astype(int)
        g3 = np.where(gg <= 1, 0, gg - 1)                           # trace / small -> 0, moderate -> 1, large -> 2
        rules = {"two_level_existing(small|moderate)": (s_amt[m] >= t_amt).astype(int),
                 "three_level_probe": (s_amt[m] >= t_amt).astype(int) + ((s_amt[m] >= t_amt) & (s[m] >= t_f1)),
                 "three_level_existing_score_only": (s_amt[m] >= t_amt).astype(int) + (s_amt[m] >= max(t_amt_large, t_amt))}
        rng = np.random.default_rng(5)
        n = int(m.sum())
        I = [rng.choice(n, n) for _ in range(B)]
        three = {}
        for k, pr in rules.items():
            hit = (pr == g3).astype(float)
            three[k] = {"acc3": float(hit.mean()),
                        "ci": [float(np.percentile([hit[i].mean() for i in I], q)) for q in (2.5, 97.5)],
                        "confusion_rows_true_cols_pred": [[int(((g3 == t) & (pr == p)).sum()) for p in range(3)]
                                                          for t in range(3)]}
        h1 = (rules["three_level_probe"] == g3).astype(float)
        for k in ("two_level_existing(small|moderate)", "three_level_existing_score_only"):
            h0 = (rules[k] == g3).astype(float)
            three[f"three_level_probe-{k}"] = {
                "delta": float(h1.mean() - h0.mean()),
                "ci": [float(np.percentile([h1[i].mean() - h0[i].mean() for i in I], q)) for q in (2.5, 97.5)]}
        p = s[m] >= t_f1
        res[sp]["probe"].update(thr=t_f1, sens=float(p[y[m] == 1].mean()), spec=float((~p)[y[m] == 0].mean()),
                                precision=float((p & (y[m] == 1)).sum() / max(p.sum(), 1)))
        res[sp]["three_level"] = three
        log(f"  {sp}: n {res[sp]['_n']}")
        for k in ("probe", "asc_amount", "owl"):
            log(f"    {k}: AUROC {res[sp][k]['auroc']:.4f} [{res[sp][k]['ci'][0]:.4f}, {res[sp][k]['ci'][1]:.4f}]")
        for k in ("probe-asc_amount", "probe-owl"):
            log(f"    {k}: {res[sp][k]['delta']:+.4f} [{res[sp][k]['ci'][0]:+.4f}, {res[sp][k]['ci'][1]:+.4f}]")
        log(f"    three-level: {json.dumps(three)}")
    d = res["val"]["probe-asc_amount"]
    dec = {"add_to_prompt": bool(d["ci"][0] > 0),
           "rule": "val paired AUROC difference (probe - existing asc_amount score, large vs not large) has a 95% CI "
                   "lower bound > 0; fixed before val was read",
           "val_paired_delta": d}
    log(f"  decision: {json.dumps(dec)}")

    f = lambda v: "" if np.isnan(v) else f"{v:.5f}"             # noqa: E731
    with open(out / "probe_scores_ascites3.csv", "w") as fo:
        w = csv.writer(fo)
        w.writerow(["case_id", "split", KEY])
        for n_ in range(len(ids)):
            w.writerow([ids[n_], split[n_], f(s[n_])])
    uc = json.load(open(a.cue_json))                            # peer / earlier cue file, read-only; copied + one key
    assert set(uc["labels_train"]) == set(ids[tr])
    for n_ in np.where(tr)[0]:
        uc["labels_train"][ids[n_]][KEY] = None if np.isnan(y[n_]) else int(y[n_])
    uc["thresholds"][KEY] = t_f1
    uc["pools"][KEY] = pools
    uc["grades"][KEY] = {"disease": "ascites", "score": "P(large | ascites reported)",
                         "label": "1 = report amount large, 0 = trace / small / moderate, null = ascites not reported "
                                  "positive or amount not written"}
    (out / "unit_cue_ascites3.json").write_text(json.dumps(uc))
    chk = {"counts": counts, "fit_rows": n_fit, "test_rows_read": 0, "splits_in_scores": sorted(set(split)),
           "label_matches_peer_asc_amount": True, "pillar_probe": info,
           "thresholds": {KEY: t_f1, "asc_amount": t_amt, "asc_amount_as_large_train_oof_f1": t_amt_large},
           "pool_sizes": {k: len(v) for k, v in pools.items()},
           "coverage": {sp: {"cases": int(m.sum()), "scored": int((m & ~np.isnan(s)).sum()),
                             "labelled": int((m & ~np.isnan(y)).sum())} for sp, m in (("train", tr), ("val", va))},
           "large_vs_not": res, "decision": dec, "bootstrap": B,
           "note": "aggregates only; threshold / pools from train OOF; val reported once, never used for tuning"}
    (out / "check.json").write_text(json.dumps(chk, indent=1))
    log(f"[{time.strftime('%m-%d %H:%M')}] wrote {out}/probe_scores_ascites3.csv, unit_cue_ascites3.json, check.json")


if __name__ == "__main__":
    main()
