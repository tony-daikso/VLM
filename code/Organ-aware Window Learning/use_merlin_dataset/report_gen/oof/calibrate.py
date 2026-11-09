"""Platt-calibrate each disease score on train OOF (label = OWL binary label), apply to train OOF and val.
Writes out/scores_trval_prob.csv (study_id, split, <d> = calibrated probability 0-1, NaN where no score) and
out/calib.json (a, b per disease). Fitted on train only; val/test never used for fitting."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
S = pd.read_csv("out/scores_trval.csv")
lab = pd.read_csv("/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best/work/base/labels14_trval.csv")
D = [c for c in S.columns if c not in ("study_id", "split")]
L = S[["study_id"]].merge(lab, on="study_id", how="left")
P = S[["study_id", "split"]].copy(); cal = {}
for d in D:
    m = (S.split == "train").values & S[d].notna().values & L[d].notna().values
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(S.loc[m, [d]].values, L.loc[m, d].values.astype(int))
    ok = S[d].notna().values; p = np.full(len(S), np.nan)
    p[ok] = lr.predict_proba(S.loc[ok, [d]].values)[:, 1]; P[d] = p
    cal[d] = dict(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]), prev_train=float(L.loc[m, d].mean()))
    for sp in ("train", "val"):
        k = (S.split == sp).values & ok & L[d].notna().values
        y = L.loc[k, d].values; pp = p[k]
        cal[d][f"brier_{sp}"] = float(np.mean((pp - y) ** 2)); cal[d][f"mean_p_{sp}"] = float(pp.mean()); cal[d][f"prev_{sp}"] = float(y.mean())
P.to_csv("out/scores_trval_prob.csv", index=False, float_format="%.5f")
json.dump(cal, open("out/calib.json", "w"), indent=1)
for d in D: print(f"{d:30s} prev tr {cal[d]['prev_train']:.3f} meanp val {cal[d]['mean_p_val']:.3f} prev val {cal[d]['prev_val']:.3f} brier val {cal[d]['brier_val']:.4f}")
