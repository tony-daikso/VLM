"""Age-shortcut diagnostics on a split (default val) for any set of probe runs.

Per finding and run: overall AUROC; age-matched AUROC (each positive vs negatives within +-tol years);
Spearman(score, age) among negatives; and the same for age alone as reference.
usage: python age_shortcut.py --config C --runs std2/E0 std2/E3_late trap/E0_trap trap/E3_late_trap [--split val]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import load_cfg
from probe import auroc


def matched_auc(y, p, age, tol):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    num = den = 0.0
    for i in pos:
        nb = neg[np.abs(age[neg] - age[i]) <= tol]
        if len(nb):
            num += (p[i] > p[nb]).sum() + 0.5 * (p[i] == p[nb]).sum()
            den += len(nb)
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", nargs="+", required=True, help="<results-subdir>/<experiment>")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    res = cfg["out"] / "results"
    lab = pd.read_csv(cfg["labels_file"]).set_index("study_id")
    meta = pd.read_csv(cfg["metadata_file"]).drop_duplicates("study id").set_index("study id")
    rows = []
    for run in a.runs:
        z = np.load(res / f"{run}.npz", allow_pickle=True)
        ids, S, findings = list(z["ids"]), z["split"], list(z["findings"])
        prob = z["prob"].mean(0)
        age_all = meta.reindex(ids)["Age"].values.astype(float)
        sel = (S == a.split) & ~np.isnan(age_all)
        for fi, f in enumerate(findings):
            y = lab.loc[np.array(ids)[sel], f].values
            ok = ~np.isnan(y)
            yy, pp, ag = y[ok], prob[sel][ok, fi], age_all[sel][ok]
            neg = yy == 0
            rows.append({"run": run, "finding": f, "n_pos": int(yy.sum()),
                         "auroc": auroc(yy, pp), "age_matched_auroc": matched_auc(yy, pp, ag, a.tol),
                         "spearman_score_age_neg": spearmanr(pp[neg], ag[neg])[0],
                         "age_only_auroc": auroc(yy, ag), "age_only_matched": matched_auc(yy, ag, ag, a.tol)})
    df = pd.DataFrame(rows)
    out = a.out or (res / f"age_shortcut_{a.split}.csv")
    df.to_csv(out, index=False)
    piv = df.pivot_table(index="finding", columns="run", values=["auroc", "age_matched_auroc", "spearman_score_age_neg"])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(piv.round(3).to_string())
    print("\nmeans over findings:")
    print(df.groupby("run")[["auroc", "age_matched_auroc", "spearman_score_age_neg"]].mean().round(4).to_string())
    print("\nage alone:", df.drop_duplicates("finding")[["finding", "age_only_auroc", "age_only_matched"]].round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
