"""Fracture analysis with geometry/HU features vs organ embeddings (val only; test is never touched).

Models (all fit on train, scored on val):
  feat_lr   logistic regression on standardized summary features (+ missing indicators)
  feat_gb   HistGradientBoosting on all per-level + summary features (NaN-native)
  emb       the embedding probe's fracture probability (e.g. E3_late), as given
  emb+feat  HistGradientBoosting on features + embedding logit
Targets: 'fracture' (any site) and 'compression' (vertebral compression = yes; other fractures excluded from training).
Report: overall AUROC, age-matched AUROC (+-2 y), and AUROC per fracture subgroup (each vs all negatives).
usage: python fracture_eval.py --features f1.csv f2.csv ... --labels-jsonl L --split-csv S --metadata M
                               [--emb-npz results/std/E3_late.npz] --out-md report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from probe import auroc  # noqa: E402

SUMMARY = ["n_levels", "min_ant_post_ratio", "min_height_nb_ratio", "max_height_loss", "min_vol_nb_ratio",
           "L1_hu", "T12_L2_hu", "min_body_hu_p10"]
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}


def load_labels(path):
    rows = {}
    for l in open(path):
        r = json.loads(l)
        a = r["answer"]["fracture"]
        d = a.get("detail", {}) or {}
        cat = a["category"]
        rows[r["study_id"]] = {
            "y": 1.0 if cat in POS else 0.0 if cat in NEG else np.nan,
            "compression": d.get("vertebral_compression") == "yes",
            "sites": d.get("sites", []) or [],
            "acute": d.get("acute", "not_stated"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--labels-jsonl", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--emb-npz", default=None)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    feat = pd.concat([pd.read_csv(f) for f in a.features]).drop_duplicates("id", keep="last").set_index("id")
    lab = load_labels(a.labels_jsonl)
    split = pd.read_csv(a.split_csv).set_index("study_id")["split"]
    age = pd.read_csv(a.metadata).drop_duplicates("study id").set_index("study id")["Age"]
    df = feat.join(lab, how="inner").join(split.rename("split"), how="inner").join(age.rename("age"))
    if a.emb_npz:
        z = np.load(a.emb_npz, allow_pickle=True)
        fi = list(z["findings"]).index("fracture")
        p = pd.Series(z["prob"].mean(0)[:, fi], index=list(z["ids"]))
        df = df.join(np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).rename("emb_logit"))
    df = df[~df["y"].isna()]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    allf = [c for c in feat.columns]

    def fit_predict(target, train_rows, model, cols):
        Xtr, ytr = train_rows[cols], target(train_rows)
        keep = ~ytr.isna()
        Xtr, ytr = Xtr[keep], ytr[keep]
        if model == "lr":
            med = Xtr.median()
            def prep(X):
                M = X.isna().astype(float).add_suffix("_na")
                return pd.concat([X.fillna(med), M], axis=1).values
            sc = StandardScaler().fit(prep(Xtr))
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(prep(Xtr)), ytr)
            return clf.predict_proba(sc.transform(prep(va[cols])))[:, 1]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(Xtr.values, ytr.values)
        return clf.predict_proba(va[cols].values)[:, 1]

    targets = {
        "fracture": lambda d: d["y"],
        "compression": lambda d: pd.Series(np.where(d["y"] == 0, 0.0, np.where(d["compression"], 1.0, np.nan)), index=d.index),
    }
    preds = {}
    for tname, tfun in targets.items():
        preds[f"{tname}:feat_lr"] = fit_predict(tfun, tr, "lr", SUMMARY)
        preds[f"{tname}:feat_gb"] = fit_predict(tfun, tr, "gb", allf)
        if "emb_logit" in df and df["emb_logit"].notna().any():
            trm = tr[tr["emb_logit"].notna()]
            preds[f"{tname}:emb+feat"] = fit_predict(tfun, trm, "gb", allf + ["emb_logit"])
    if "emb_logit" in df:
        preds["emb"] = va["emb_logit"].values
    preds["age_only"] = va["age"].values

    neg = va["y"] == 0
    groups = {
        "全部骨折": va["y"] == 1,
        "椎體壓迫性": (va["y"] == 1) & va["compression"],
        "只有脊椎": (va["y"] == 1) & va["sites"].apply(lambda s: s == ["vertebra"]),
        "只有肋骨": (va["y"] == 1) & va["sites"].apply(lambda s: s == ["rib"]),
        "骨盆/薦椎/髖": (va["y"] == 1) & va["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))),
        "急性(含混合)": (va["y"] == 1) & va["acute"].isin(["yes", "mixed"]),
        "陳舊": (va["y"] == 1) & (va["acute"] == "no"),
    }
    L = [f"# 骨折：幾何/HU 特徵 vs embedding（val，N={len(va)}，陰性 {int(neg.sum())}）\n",
         "| 模型 | " + " | ".join(f"{g}（{int(m.sum())}）" for g, m in groups.items()) + " | 年齡配對（全部骨折） |",
         "|---" * (len(groups) + 2) + "|"]
    agev = va["age"].values
    for name, p in preds.items():
        p = np.asarray(p, float)
        cells = []
        for g, m in groups.items():
            sel = (m | neg).values & ~np.isnan(p)
            cells.append(f"{auroc(m.values[sel].astype(float), p[sel]):.3f}")
        sel = (groups["全部骨折"] | neg).values & ~np.isnan(p) & ~np.isnan(agev)
        mat = matched_auc(groups["全部骨折"].values[sel].astype(int), p[sel], agev[sel], 2.0)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {mat:.3f} |")
    L += ["", "特徵涵蓋率（val）：", f"- 有 ≥1 節 T8–L5 椎體：{(va['n_levels'] > 0).mean():.3f}；平均節數 {va['n_levels'].mean():.1f}",
          f"- L1 HU 中位數：陽性 {va.loc[va.y == 1, 'L1_hu'].median():.0f} / 陰性 {va.loc[neg, 'L1_hu'].median():.0f}",
          f"- 最大高度損失中位數：壓迫性陽性 {va.loc[groups['椎體壓迫性'], 'max_height_loss'].median():.3f} / 陰性 {va.loc[neg, 'max_height_loss'].median():.3f}"]
    Path(a.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
