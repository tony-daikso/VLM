#!/usr/bin/env python3
"""Method 1 on VAL: assembly-decision metrics (what the template wrote, without Qwen back-extraction).
Generated positive = pos or verify (a positive sentence was written); generated negative = a negative sentence was
written (decision neg and the disease is not a silent-negative one); silent / omit = nothing written.
  sens = TP / report positives ; spec = (report negatives NOT given a positive sentence) / report negatives
  critical error = report positive but a NEGATIVE sentence was written (the key safety metric)
Labels: definition B (primary; negatives = explicit no/probably_no) and definition A (not_mentioned = negative).
Measured attributes: AAA diameter |gen - report| <= 0.3 cm, spleen length |gen - report| <= 1.0 cm, on report-positive
cases where the report states the value and method 1 wrote the positive sentence with a measurement.
Outputs: method1/val_decision_metrics.csv, method1/val_attr_measured.csv"""
import json

import numpy as np
import pandas as pd

from m1_common import D14, OUT, ZH, defa, defb, load_labels


def run(var, L, silent):
    dec = pd.read_csv(OUT / f"val_decisions_{var}.csv", index_col=0)
    rows = []
    for lbl, f in (("defB", defb), ("defA", defa)):
        for d in D14:
            y = f(L[d]).reindex(dec.index).values; x = dec[d].values
            gpos = np.isin(x, ["pos", "verify"]); gneg = (x == "neg") & (not silent[d])
            k = ~np.isnan(y); y, gpos, gneg, xx = y[k], gpos[k], gneg[k], x[k]
            tp = int((gpos & (y == 1)).sum()); fp = int((gpos & (y == 0)).sum()); fn = int((~gpos & (y == 1)).sum())
            tn = int((~gpos & (y == 0)).sum())
            rows.append(dict(variant=var, label=lbl, disease=d, 疾病=ZH[d], N=int(len(y)), n_pos=int((y == 1).sum()),
                             n_neg=int((y == 0).sum()), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                             ppv=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                             crit_err=int((gneg & (y == 1)).sum()), crit_rate=float((gneg & (y == 1)).sum() / max((y == 1).sum(), 1)),
                             neg_written_on_neg=float((gneg & (y == 0)).sum() / max((y == 0).sum(), 1)),
                             frac_verify=float(np.mean(xx == "verify"))))
    return rows


def attrs(var, L):
    rows = []
    D = {json.loads(l)["case_id"]: json.loads(l) for l in open(OUT / f"val_drafts_{var}.jsonl")}
    for d, key, tol, mk in (("abdominal_aortic_aneurysm", "diameter_cm", 0.3, "aaa_cm"), ("splenomegaly", "length_cm", 1.0, "spleen_cm")):
        ref = L[f"{d}__detail"].map(lambda s: json.loads(s).get(key)); y = defb(L[d])
        sel = [s for s in L.index if y.get(s) == 1 and ref.get(s) is not None and not pd.isna(ref.get(s)) and s in D]
        n_stated = len(sel); wrote = [s for s in sel if mk in D[s]["measured"]]
        err = np.array([abs(D[s]["measured"][mk] - float(ref[s])) for s in wrote])
        rows.append(dict(variant=var, disease=d, 疾病=ZH[d], attribute=key, n_report_pos_with_value=n_stated,
                         n_written=len(wrote), within_tol=float(np.mean(err <= tol)) if len(err) else np.nan,
                         within_tol_over_all=float(np.sum(err <= tol) / max(n_stated, 1)), mae=float(err.mean()) if len(err) else np.nan, tol=tol))
    return rows


def main():
    L = load_labels(("val",))
    silent = json.load(open(OUT / "sentence_bank.json"))["silent_negative"]
    R = pd.DataFrame(run("A", L, silent) + run("B", L, silent)); R.to_csv(OUT / "val_decision_metrics.csv", index=False)
    A = pd.DataFrame(attrs("A", L) + attrs("B", L)); A.to_csv(OUT / "val_attr_measured.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[R.label == "defB"].round(3).to_string(index=False)); print(A.round(3).to_string(index=False))
    for var in ("A", "B"):
        r = R[(R.label == "defB") & (R.variant == var)]
        print(var, "macro sens %.3f spec %.3f F1 %.3f | crit %d / %d pos = %.4f" % (r.sens.mean(), r.spec.mean(), r.f1.mean(),
              r.crit_err.sum(), r.n_pos.sum(), r.crit_err.sum() / r.n_pos.sum()))


if __name__ == "__main__":
    main()
