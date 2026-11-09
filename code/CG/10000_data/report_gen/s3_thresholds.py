#!/usr/bin/env python3
"""S3: per-class two-threshold rule on OWL logits, chosen on DEV only (definition B labels).

  score <  t_neg            -> negative sentence      (t_neg: sensitivity >= 0.95 on dev positives)
  score >  t_pos            -> positive sentence      (t_pos: specificity >= 0.95 on dev negatives)
  otherwise                 -> positive sentence + [please verify], no negative sentence
If t_pos < t_neg the positive rule wins (never a negative sentence above t_neg).

Rare classes (rule "rare_silent", < 20 dev positives: bowel obstruction, appendicitis, free air): with
prevalence ~1-3 %, specificity 0.95 gives a PPV of a few percent and the sens>=0.95 cut (= lowest of <10 dev
positives) leaves 30-70 % of exams in the middle zone. A definitive positive sentence or a "[please verify]"
positive on most drafts is unusable, so for these classes: score > t_pos -> positive + [please verify] (never a
definitive positive); middle zone -> no sentence at all (still never a negative statement); score < t_neg ->
negative sentence.

Small classes: allowed misses = floor(0.05 * n_pos). With n_pos < 20 this is 0, i.e. t_neg = lowest dev positive
score (dev sensitivity 1.0); such classes are flagged low_n and the exact (Clopper-Pearson) lower bound of the
dev sensitivity is reported. Same for specificity with few negatives (atherosclerosis).

Measurement guards (S4, deterministic, also fixed on dev): a negative decision is raised to [please verify] when
 - abdominal_aortic_aneurysm: auto-measured max abdominal aortic diameter >= 30 mm (the AAA definition);
 - splenomegaly: auto spleen volume > the 97.5th percentile of dev def-B-negative volumes (not FOV-truncated);
 - osteopenia: L1 HU <= 110; and a positive decision is lowered to [please verify] when L1 HU > 160.
Outputs: thresholds.csv, s3_dev_metrics.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import beta

from common import DIS, MEAS_TS, OSTEO, OUT, PROBS, ZH, defb_labels

SENS, SPEC, LOW_N = 0.95, 0.95, 20


def cp_lower(k, n, a=0.05):
    return 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))


def pick(scores, y):
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])[::-1]
    miss = int(np.floor((1 - SENS) * len(pos)))
    fp = int(np.floor((1 - SPEC) * len(neg)))
    return float(pos[miss]), float(neg[fp]), miss, fp


def decide(s, t_neg, t_pos, rule="standard"):
    if rule == "rare_silent":  # see module docstring: <20 dev positives
        return np.where(s > t_pos, "verify", np.where(s >= t_neg, "omit", "neg"))
    return np.where(s > t_pos, "pos", np.where(s >= t_neg, "verify", "neg"))


def load_meas():
    m = pd.read_csv(MEAS_TS).set_index("study_id")
    hu = pd.read_csv(OSTEO).set_index("study_id")
    m["L1_hu"] = hu["hu"]
    m["hu_level"] = hu["level_used"]
    m["hu_excl"] = hu["excl_reason"]
    m.loc[(m.spleen_trunc == 1), "spleen_ml"] = np.nan
    m.loc[(m.aorta_z_n < 10), "aorta_eqd_mm"] = np.nan
    return m


def apply_guards(dec, meas, guards):
    """dec: DataFrame EXAMID x DIS of pos/verify/neg; returns guarded copy + per-class count of changes."""
    d = dec.copy()
    m = meas.reindex(d.index)
    rules = [("abdominal_aortic_aneurysm", "neg", m.aorta_eqd_mm >= guards["aorta_mm"]),
             ("splenomegaly", "neg", m.spleen_ml > guards["spleen_ml"]),
             ("osteopenia", "neg", m.L1_hu <= 110),
             ("osteopenia", "pos", m.L1_hu > 160)]
    changed = {}
    for k, frm, cond in rules:
        hit = (d[k] == frm) & cond.fillna(False).values
        d.loc[hit, k] = "verify"
        changed[f"{k}:{frm}->verify"] = int(hit.sum())
    return d, changed


def metrics(dec, y):
    """assembly-level: positive = pos or verify; also the 'stated negative while positive' error."""
    keep = ~np.isnan(y)
    d, yy = dec[keep], y[keep]
    p = np.isin(d, ["pos", "verify"])
    tp, fn = int((p & (yy == 1)).sum()), int((~p & (yy == 1)).sum())
    tn, fp = int((~p & (yy == 0)).sum()), int((p & (yy == 0)).sum())
    return dict(N=int(keep.sum()), n_pos=int((yy == 1).sum()), n_neg=int((yy == 0).sum()),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                frac_neg=float(np.mean(dec == "neg")), frac_verify=float(np.mean(dec == "verify")),
                frac_pos=float(np.mean(dec == "pos")), frac_omit=float(np.mean(dec == "omit")),
                neg_stmt_on_pos=int(((d == "neg") & (yy == 1)).sum()),
                pos_stmt_strict_on_neg=int(((d == "pos") & (yy == 0)).sum()))


def main():
    split = pd.read_csv(OUT / "split.csv").set_index("EXAMID")
    probs = pd.read_csv(PROBS).set_index("study_id")
    lab = defb_labels().reindex(probs.index)
    meas = load_meas()
    dev = split.index[split.split == "dev"]
    rows = []
    for k in DIS:
        s, y = probs.loc[dev, k].values, lab.loc[dev, k].values
        keep = ~np.isnan(y)
        t_neg, t_pos, miss, fp = pick(s[keep], y[keep])
        npos, nneg = int((y[keep] == 1).sum()), int((y[keep] == 0).sum())
        rows.append(dict(disease=k, 疾病=ZH[k], t_neg=t_neg, t_pos=t_pos, dev_n_pos=npos, dev_n_neg=nneg,
                         allowed_miss=miss, allowed_fp=fp,
                         dev_sens_at_tneg=float(np.mean(s[keep][y[keep] == 1] >= t_neg)),
                         dev_sens_cp_lower=cp_lower(int(np.sum(s[keep][y[keep] == 1] >= t_neg)), npos),
                         dev_spec_at_tpos=float(np.mean(s[keep][y[keep] == 0] <= t_pos)),
                         low_n="low_n_pos" if npos < LOW_N else ("low_n_neg" if nneg < 100 else ""),
                         rule="rare_silent" if npos < LOW_N else "standard",
                         label="L1 HU<=110 pos / >160 neg" if k == "osteopenia" else "report def B"))
    th = pd.DataFrame(rows)

    devneg = lab.loc[dev, "splenomegaly"] == 0
    guards = {"aorta_mm": 30.0,
              "spleen_ml": float(np.nanpercentile(meas.reindex(dev[devneg.values]).spleen_ml, 97.5))}
    th.to_csv(OUT / "thresholds.csv", index=False)
    json.dump(guards, open(OUT / "guards.json", "w"), indent=1)

    dec = pd.DataFrame({k: decide(probs.loc[dev, k].values, *th.set_index("disease").loc[k, ["t_neg", "t_pos", "rule"]])
                        for k in DIS}, index=dev)
    gdec, changed = apply_guards(dec, meas, guards)
    mrows = []
    for k in DIS:
        y = lab.loc[dev, k].values
        mrows.append(dict(disease=k, 疾病=ZH[k], stage="threshold_only", **metrics(dec[k].values, y)))
        mrows.append(dict(disease=k, 疾病=ZH[k], stage="with_guards", **metrics(gdec[k].values, y)))
    M = pd.DataFrame(mrows)
    M.to_csv(OUT / "s3_dev_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(th.round(3).to_string(index=False))
    print(guards, changed)
    print(M.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
