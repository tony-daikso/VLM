"""Honest (out-of-fold) version of fracture_eval.py: the stack is fit on OOF embedding logits for TRAIN rows.

Test rows are dropped right after joining and never scored.
Models (fit on train, scored on val):
  feat_lr / feat_gb      as in fracture_eval.py
  emb+feat               HistGradientBoosting on all features + OOF emb logit
  emb+5feat_lr           logistic regression on OOF emb logit + 5 key summary features (+ missing indicators)
  emb+feat(in-sample)    [optional, --emb-npz] the old biased stack (train emb logits in-sample), for reference
  emb                    val logit from the OOF npz (mean of the K fold models)
  emb(probe)             [optional, --emb-npz] the original probe's val probability
Plus a paired bootstrap on val (resampling studies) of AUROC and age-matched AUROC differences vs emb.
usage: python fracture_eval_oof.py --features f*.csv --labels-jsonl L --split-csv S --metadata M
                                   --emb-npz-oof outputs_fracture/oof_E0.npz [--emb-npz results/std/E0.npz]
                                   --out-md report.md [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval import SUMMARY, load_labels  # noqa: E402
from probe import auroc  # noqa: E402

KEY5 = ["min_ant_post_ratio", "min_height_nb_ratio", "max_height_loss", "min_vol_nb_ratio", "L1_hu"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot_diff(y, a, b, age, tol, n_boot, seed=0):
    """Paired bootstrap over studies of AUROC(a)-AUROC(b) and age-matched AUROC(a)-AUROC(b).
    Uses count weights: AUC = c_pos^T (K o M) c_neg / c_pos^T M c_neg, M = all-ones or |age diff|<=tol."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    def K(s):
        d = s[pos][:, None] - s[neg][None, :]
        return (d > 0) + 0.5 * (d == 0)
    Ka, Kb = K(a), K(b)
    M = (np.abs(age[pos][:, None] - age[neg][None, :]) <= tol).astype(float)
    Dk, DkM = Ka - Kb, (Ka - Kb) * M
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {"auc": [], "matched": []}
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        cp, cn = c[pos], c[neg]
        den = cp.sum() * cn.sum()
        out["auc"].append(cp @ Dk @ cn / den)
        out["matched"].append(cp @ DkM @ cn / (cp @ M @ cn))
    pt = {"auc": auroc(y.astype(float), a) - auroc(y.astype(float), b),
          "matched": matched_auc(y, a, age, tol) - matched_auc(y, b, age, tol)}
    return {k: (pt[k], *np.percentile(v, [2.5, 97.5]), float(np.mean(np.asarray(v) <= 0))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--labels-jsonl", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--emb-npz-oof", required=True)
    ap.add_argument("--emb-npz", default=None, help="original probe npz (in-sample on train) for reference rows")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    feat = pd.concat([pd.read_csv(f) for f in a.features]).drop_duplicates("id", keep="last").set_index("id")
    lab = load_labels(a.labels_jsonl)
    split = pd.read_csv(a.split_csv).set_index("study_id")["split"]
    age = pd.read_csv(a.metadata).drop_duplicates("study id").set_index("study id")["Age"]
    df = feat.join(lab, how="inner").join(split.rename("split"), how="inner").join(age.rename("age"))
    df = df[df["split"].isin(["train", "val"])]                     # test never enters anything below
    z = np.load(a.emb_npz_oof, allow_pickle=True)
    assert not np.isin(z["split"], ["test"]).any()
    df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("emb_logit"))
    if a.emb_npz:
        zi = np.load(a.emb_npz, allow_pickle=True)
        fi = list(zi["findings"]).index("fracture")
        ok = zi["split"] != "test"
        df = df.join(pd.Series(logit(zi["prob"].mean(0)[ok, fi]), index=np.array(zi["ids"])[ok]).rename("emb_insample"))
    df = df[~df["y"].isna()]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    allf = list(feat.columns)

    def fit_predict(target, train_rows, model, cols):
        Xtr, ytr = train_rows[cols], target(train_rows)
        keep = ~ytr.isna()
        Xtr, ytr = Xtr[keep], ytr[keep]
        if model == "lr":
            med = Xtr.median()
            def prep(X):
                Mi = X.isna().astype(float).add_suffix("_na")
                return pd.concat([X.fillna(med), Mi], axis=1).values
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
    trm = tr[tr["emb_logit"].notna()]
    preds = {}
    for tname, tfun in targets.items():
        preds[f"{tname}:feat_lr"] = fit_predict(tfun, tr, "lr", SUMMARY)
        preds[f"{tname}:feat_gb"] = fit_predict(tfun, tr, "gb", allf)
        preds[f"{tname}:emb+feat(OOF)"] = fit_predict(tfun, trm, "gb", allf + ["emb_logit"])
        preds[f"{tname}:emb+5feat_lr(OOF)"] = fit_predict(tfun, trm, "lr", ["emb_logit"] + KEY5)
        if "emb_insample" in df:
            tri = tr[tr["emb_insample"].notna()].drop(columns="emb_logit").rename(columns={"emb_insample": "emb_logit"})
            vv = va
            va = va.drop(columns="emb_logit").rename(columns={"emb_insample": "emb_logit"})
            preds[f"{tname}:emb+feat(in-sample,舊)"] = fit_predict(tfun, tri, "gb", allf + ["emb_logit"])
            va = vv
    preds["emb(OOF 5-fold 平均)"] = va["emb_logit"].values
    if "emb_insample" in df:
        preds["emb(原 probe)"] = va["emb_insample"].values
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
    oofname = Path(a.emb_npz_oof).stem
    L = [f"# 骨折（OOF 誠實版，emb={oofname}）：val N={len(va)}，陰性 {int(neg.sum())}；train N={len(tr)}\n",
         "| 模型 | " + " | ".join(f"{g}（{int(m.sum())}）" for g, m in groups.items()) + " | 年齡配對（全部骨折） | 年齡配對（壓迫性） |",
         "|---" * (len(groups) + 3) + "|"]
    agev = va["age"].values
    for name, p in preds.items():
        p = np.asarray(p, float)
        cells = []
        for g, m in groups.items():
            sel = (m | neg).values & ~np.isnan(p)
            cells.append(f"{auroc(m.values[sel].astype(float), p[sel]):.3f}")
        for g in ("全部骨折", "椎體壓迫性"):
            sel = (groups[g] | neg).values & ~np.isnan(p) & ~np.isnan(agev)
            cells.append(f"{matched_auc(groups[g].values[sel].astype(int), p[sel], agev[sel], 2.0):.3f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    base = np.asarray(preds["emb(OOF 5-fold 平均)"], float)
    L += ["", f"## Paired bootstrap（val，{a.n_boot} 次，重抽 study；差值 = 模型 − emb(OOF)；括號為 95% CI；p = P(差≤0)）\n",
          "| 比較 | 評估族群 | ΔAUROC | 95% CI | p | Δ年齡配對 AUROC | 95% CI | p |", "|---|---|---|---|---|---|---|---|"]
    for mname in ["fracture:emb+feat(OOF)", "compression:emb+feat(OOF)", "fracture:emb+5feat_lr(OOF)", "compression:emb+5feat_lr(OOF)"]:
        for g in ("全部骨折", "椎體壓迫性"):
            sel = ((groups[g] | neg).values & ~np.isnan(agev))
            r = boot_diff(groups[g].values[sel].astype(int), np.asarray(preds[mname], float)[sel], base[sel], agev[sel], 2.0, a.n_boot)
            d, lo, hi, p = r["auc"]; dm, lom, him, pm = r["matched"]
            L.append(f"| {mname} | {g} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {dm:+.3f} | [{lom:+.3f}, {him:+.3f}] | {pm:.3f} |")
    L += ["", f"5 個關鍵特徵（emb+5feat_lr）：{', '.join(KEY5)}"]
    Path(a.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
