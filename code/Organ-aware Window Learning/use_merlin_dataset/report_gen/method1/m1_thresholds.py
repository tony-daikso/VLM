#!/usr/bin/env python3
"""Method 1, step S3: per-disease thresholds on TRAIN OOF scores only.

Threshold label = definition A (yes/probably_yes vs no/probably_no/not_mentioned; unclear excluded), i.e. the label the
OWL heads were trained on. Definition B (explicit negatives only) leaves too few negatives for some diseases (train:
osteopenia 43, atherosclerosis 666), so cuts chosen on B degenerate ("call everyone positive"). Positives are the
same under A and B, so t_neg (sensitivity >= 0.95) is identical; metrics are reported under both definitions.

Variant A (CG-style, two thresholds, safety first):
  p <  t_neg -> negative sentence   (t_neg: train-OOF sensitivity >= 0.95)
  p >  t_pos -> positive sentence   (t_pos: train-OOF specificity >= 0.95)
  else       -> positive sentence + [please verify]   (never a negative sentence)
Variant B (single threshold, definitive text like method 4 will produce): t_f1 = train-OOF F1-optimal cut;
  p >= t_f1 -> positive sentence, else negative sentence.
Studies without a score (n/a) -> no sentence ("omit").
Outputs: method1/thresholds.csv, method1/train_decision_metrics.csv
"""
import numpy as np
import pandas as pd

from m1_common import D14, OUT, PROB, ZH, defa, defb, load_labels

SENS, SPEC = 0.95, 0.95


def pick(s, y):
    pos = np.sort(s[y == 1]); neg = np.sort(s[y == 0])[::-1]
    t_neg = float(pos[int(np.floor((1 - SENS) * len(pos)))])
    t_pos = float(neg[int(np.floor((1 - SPEC) * len(neg)))])
    # F1-optimal single cut
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + ys.sum()); k = int(np.argmax(f1))
    return t_neg, t_pos, float(s[o][k]), float(f1[k])


def decide_a(p, t_neg, t_pos):
    d = np.where(p > t_pos, "pos", np.where(p >= t_neg, "verify", "neg"))
    return np.where(np.isnan(p), "omit", d)


def decide_b(p, t_f1):
    return np.where(np.isnan(p), "omit", np.where(p >= t_f1, "pos", "neg"))


def dec_metrics(d, y):
    k = ~np.isnan(y); d, y = d[k], y[k]
    pos = np.isin(d, ["pos", "verify"])
    tp = int((pos & (y == 1)).sum()); fn = int((~pos & (y == 1)).sum())
    tn = int((~pos & (y == 0)).sum()); fp = int((pos & (y == 0)).sum())
    return dict(N=int(len(y)), n_pos=int((y == 1).sum()), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                crit_err=int(((d == "neg") & (y == 1)).sum()), crit_rate=float(((d == "neg") & (y == 1)).sum() / max((y == 1).sum(), 1)),
                frac_neg=float(np.mean(d == "neg")), frac_verify=float(np.mean(d == "verify")),
                frac_pos=float(np.mean(d == "pos")), frac_omit=float(np.mean(d == "omit")))


def main():
    P = pd.read_csv(PROB).set_index("study_id")
    L = load_labels(("train",))
    tr = P.index[P.split == "train"]
    rows, mrows = [], []
    for d in D14:
        p = P.loc[tr, d].values; ya = defa(L[d].reindex(tr)).values; y = defb(L[d].reindex(tr)).values
        k = ~np.isnan(p) & ~np.isnan(ya)
        t_neg, t_pos, t_f1, f1 = pick(p[k], ya[k])
        rows.append(dict(disease=d, 疾病=ZH[d], t_neg=t_neg, t_pos=t_pos, t_f1=t_f1, train_f1_at_tf1=f1,
                         train_n_pos=int((ya[k] == 1).sum()), train_n_neg_defA=int((ya[k] == 0).sum()),
                         train_n_neg_defB=int((y == 0).sum()), train_n_unclear=int(np.isnan(ya).sum())))
        for lbl, yy in (("defB", y), ("defA", ya)):
            mrows.append(dict(disease=d, 疾病=ZH[d], variant="A_two_thr", label=lbl, **dec_metrics(decide_a(p, t_neg, t_pos), yy)))
            mrows.append(dict(disease=d, 疾病=ZH[d], variant="B_f1", label=lbl, **dec_metrics(decide_b(p, t_f1), yy)))
    th = pd.DataFrame(rows); th.to_csv(OUT / "thresholds.csv", index=False)
    M = pd.DataFrame(mrows); M.to_csv(OUT / "train_decision_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(th.round(4).to_string(index=False)); print(M.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
