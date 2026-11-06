"""Single-finding (default: fracture) val comparison of finetuned vs frozen runs on identical val ids.

Per run: AUROC, age-matched AUROC (+-tol y), Spearman(score, age) on negatives, subgroup AUROCs from the detail
labels (labels_disease14_all.jsonl, e.g. fracture.detail.vertebral_compression == yes vs all negatives), and
paired bootstrap (by study) differences vs reference runs. VAL ONLY: every run must contain only split == val.
usage: python ft_frac_eval.py --config C --runs finetune_frac/E5_ft2_frac_trap finetune_frac/E0_final ... \
          --refs finetune_frac/E0_final finetune_frac/E3_late_final [--finding fracture --detail vertebral_compression]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from age_shortcut import matched_auc
from common import load_cfg
from probe import auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--finding", default="fracture")
    ap.add_argument("--detail", default="vertebral_compression")
    ap.add_argument("--detail-file", default=None)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-matched", action="store_true", help="also bootstrap the age-matched AUROC (slow)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    res = cfg["out"] / "results"
    P, ids0 = {}, None
    for r in a.runs:
        z = np.load(res / f"{r}.npz", allow_pickle=True)
        assert (z["split"] == "val").all(), f"{r}: non-val rows present"
        ids = list(z["ids"])
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, f"{r}: ids differ; run finetune.py --align-ref first"
        P[r] = z["prob"].mean(0)[:, list(z["findings"]).index(a.finding)]
    ids = np.array(ids0)
    y = pd.read_csv(cfg["labels_file"]).set_index("study_id").loc[ids, a.finding].values.astype(float)
    age = pd.read_csv(cfg["metadata_file"]).drop_duplicates("study id").set_index("study id").reindex(ids)["Age"].values.astype(float)
    det = {}
    dfile = a.detail_file or str(cfg["labels_file"]).replace("labels_disease14.csv", "labels_disease14_all.jsonl")
    for line in open(dfile):
        d = json.loads(line)
        if d.get("split") == "val":
            det[d["study_id"]] = str(d["answer"].get(a.finding, {}).get("detail", {}).get(a.detail, "na"))
    dv = np.array([det.get(i, "na") for i in ids])
    ok = ~np.isnan(y)
    sub_yes = ok & ((y == 0) | ((y == 1) & (dv == "yes")))
    sub_no = ok & ((y == 0) | ((y == 1) & (dv != "yes")))
    agok = ok & ~np.isnan(age)

    def metrics(p, idx=None, matched=True):
        idx = np.arange(len(y)) if idx is None else idx
        yy, pp, aa = y[idx], p[idx], age[idx]
        m, ma = ~np.isnan(yy), ~np.isnan(yy) & ~np.isnan(aa)
        sy = sub_yes[idx]; sn = sub_no[idx]
        neg = ma & (yy == 0)
        out = {"auroc": auroc(yy[m], pp[m]),
               f"auroc_{a.detail}_yes": auroc(yy[sy], pp[sy]),
               f"auroc_{a.detail}_not_yes": auroc(yy[sn], pp[sn])}
        if matched:
            out["age_matched"] = matched_auc(yy[ma], pp[ma], aa[ma], a.tol)
            out["spearman_neg"] = spearmanr(pp[neg], aa[neg])[0]
        return out

    rows = []
    for r in a.runs:
        rows.append({"run": r, **metrics(P[r])})
    rows.append({"run": "age_only", **metrics(age)})
    df = pd.DataFrame(rows).set_index("run")
    n_pos = int(np.nansum(y)); n_c = int(((y == 1) & (dv == "yes")).sum())
    print(f"val n={int(ok.sum())} pos={n_pos} ({a.detail}=yes: {n_c})")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    rng = np.random.default_rng(0)
    keys = ["auroc", f"auroc_{a.detail}_yes"] + (["age_matched"] if a.boot_matched else [])
    allr = list(dict.fromkeys(a.runs + a.refs))
    point = {r: metrics(P[r]) for r in allr}
    B = {r: {k: [] for k in keys} for r in allr}
    for _ in range(a.n_boot):
        b = rng.integers(0, len(y), len(y))
        for r in allr:
            mb = metrics(P[r], b, matched=a.boot_matched)
            for k in keys:
                B[r][k].append(mb[k])
    diffs = []
    for r in a.runs:
        for ref in a.refs:
            if r == ref:
                continue
            for k in keys:
                d = np.array(B[r][k]) - np.array(B[ref][k])
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                diffs.append({"run": r, "ref": ref, "metric": k, "diff": point[r][k] - point[ref][k],
                              "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
        if "age_matched" not in keys:
            for ref in a.refs:
                if r != ref:
                    diffs.append({"run": r, "ref": ref, "metric": "age_matched (point only)",
                                  "diff": point[r]["age_matched"] - point[ref]["age_matched"],
                                  "ci_lo": np.nan, "ci_hi": np.nan, "excludes_0": None})
    if diffs:
        print("\npaired bootstrap (by study), 95% CI:")
        print(pd.DataFrame(diffs).round(4).to_string(index=False))
    out = a.out or (res / f"ft_{a.finding}_eval_val.json")
    json.dump({"n": int(ok.sum()), "n_pos": n_pos, "n_detail_yes": n_c, "metrics": df.reset_index().to_dict("records"),
               "paired": diffs}, open(out, "w"), indent=1, default=float)
    print("->", out)


if __name__ == "__main__":
    main()
