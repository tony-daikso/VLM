"""Exp D: finetuned fracture logits vs frozen E0 / OOF stack on VAL, plus stacking the finetuned logit.

Population = the OOF-stack population of fracture_eval_oof.py (val studies with summary features and a fracture
label), so numbers are directly comparable with fracture_eval_oof_E3late.md (emb+feat(OOF) = 0.788).
Rows:
  E0(probe) / E3_late(probe)   frozen full-data probes (results/std), val probabilities
  stack:emb+feat(OOF)          HistGB on summary features + OOF E3_late logit, fit on train (current best)
  E5_<run>                     finetuned model, val probabilities
  stack+E5_<run>(train in-sample)   same HistGB + finetuned logit; train logits are IN-SAMPLE (biased reference)
  valCV:stack+E5_<run>         5-fold CV inside val: logistic regression on [logit(stack), finetuned logit];
                               every val study scored by a model not fit on it (honest, but uses val for fitting)
  valCV:stack                  same CV with logit(stack) only (control for the CV recalibration)
Metrics per row: AUROC per subgroup (all, compression, vertebra-only, rib-only, pelvis/sacrum/hip, acute, old),
age-matched AUROC (+-2 y), Spearman(score, age) on negatives. Paired bootstrap (studies) vs E0 and vs the stack.
Test rows are dropped right after joining and never scored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval import load_labels  # noqa: E402
from fracture_eval_oof import boot_diff, logit  # noqa: E402
from probe import auroc  # noqa: E402


def frac_prob(path):
    z = np.load(path, allow_pickle=True)
    fi = list(z["findings"]).index("fracture")
    ok = z["split"] != "test"
    s = pd.Series(z["prob"].mean(0)[ok, fi], index=np.array(z["ids"])[ok])
    return s[~s.index.duplicated()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--labels-jsonl", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--oof-npz", required=True, help="oof_E3late.npz (OOF train logits + val fold-mean)")
    ap.add_argument("--probe-e0", required=True); ap.add_argument("--probe-e3late", required=True)
    ap.add_argument("--ft", nargs="+", required=True, help="name=val_npz[,train_npz]")
    ap.add_argument("--bone-oof", default=None, help="oof_bone_*.npz (exp B); adds the new-best stack and restricts "
                    "the population to studies with a bone logit (exp B common set)")
    ap.add_argument("--bone-ens", default=None, help="exp F bone_ens.npz (ids/split/logit): val-level fixed equal-weight "
                    "average of val-z-scored logits with each finetune (no weights tuned on val)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    feat = pd.concat([pd.read_csv(f) for f in a.features]).drop_duplicates("id", keep="last").set_index("id")
    lab = load_labels(a.labels_jsonl)
    split = pd.read_csv(a.split_csv).set_index("study_id")["split"]
    age = pd.read_csv(a.metadata).drop_duplicates("study id").set_index("study id")["Age"]
    df = feat.join(lab, how="inner").join(split.rename("split"), how="inner").join(age.rename("age"))
    df = df[df["split"].isin(["train", "val"])]                     # test never enters anything below
    z = np.load(a.oof_npz, allow_pickle=True)
    assert not np.isin(z["split"], ["test"]).any()
    df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("emb_logit"))
    if a.bone_oof:
        zb = np.load(a.bone_oof, allow_pickle=True)
        assert not np.isin(zb["split"], ["test"]).any()
        df = df.join(pd.Series(zb["logit"], index=list(zb["ids"])).rename("bone_logit"))
        df = df[df["bone_logit"].notna()]
    if a.bone_ens:
        ze = np.load(a.bone_ens, allow_pickle=True)
        assert not np.isin(ze["split"], ["test"]).any()
        df = df.join(pd.Series(ze["logit"], index=list(ze["ids"])).rename("bone_ens"))
    df = df.join(frac_prob(a.probe_e0).rename("p_e0")).join(frac_prob(a.probe_e3late).rename("p_e3late"))
    fts = {}
    for spec in a.ft:
        name, paths = spec.split("=")
        pv, _, ptr = paths.partition(",")
        s = frac_prob(pv)
        if ptr:
            st = frac_prob(ptr)
            s = pd.concat([s, st[~st.index.isin(s.index)]])
        df = df.join(logit(s.clip(1e-6, 1 - 1e-6)).rename(f"ft_{name}"))
        fts[name] = bool(ptr)
    df = df[~df["y"].isna()]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    allf = list(feat.columns)
    for name in fts:
        assert va[f"ft_{name}"].notna().all(), f"{name}: val coverage incomplete"

    def gb(cols, rows):
        rows = rows[rows[cols[-1]].notna()]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(rows[cols].values, rows["y"].values)
        return clf.predict_proba(va[cols].values)[:, 1]

    def val_cv(X):
        out = np.zeros(len(va))
        y = va["y"].values.astype(int)
        for fit, hold in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[fit], y[fit])
            out[hold] = clf.decision_function(X[hold])
        return out

    preds = {"E0(probe)": va["p_e0"].values, "E3_late(probe)": va["p_e3late"].values}
    stack = gb(allf + ["emb_logit"], tr)
    preds["stack:emb+feat(OOF)"] = stack
    ls = logit(stack)
    preds["valCV:stack"] = val_cv(ls[:, None])
    best = "stack:emb+feat(OOF)"
    if a.bone_oof:
        preds["bone_OOF"] = va["bone_logit"].values
        stack = gb(allf + ["emb_logit", "bone_logit"], tr)
        best = "stackB:emb+bone+feat(OOF)"
        preds[best] = stack
        ls = logit(stack)
        preds["valCV:stackB"] = val_cv(ls[:, None])
    for name, has_tr in fts.items():
        c = f"ft_{name}"
        preds[f"E5_{name}"] = va[c].values
        base_cols = allf + ["emb_logit"] + (["bone_logit"] if a.bone_oof else [])
        tag = "stackB" if a.bone_oof else "stack"
        if has_tr:
            preds[f"{tag}+E5_{name}(train in-sample)"] = gb(base_cols + [c], tr)
        preds[f"valCV:{tag}+E5_{name}"] = val_cv(np.c_[ls, va[c].values])
    if a.bone_ens:
        assert va["bone_ens"].notna().all()
        zs = lambda v: (v - v.mean()) / v.std()
        preds["bone_ens(F1)"] = va["bone_ens"].values
        for name in fts:
            preds[f"avg(bone_ens,E5_{name})"] = 0.5 * zs(va["bone_ens"].values) + 0.5 * zs(va[f"ft_{name}"].values)
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
    agev = va["age"].values
    L = [f"# 實驗 D（finetune）骨折 val：N={len(va)}，陰性 {int(neg.sum())}；stack 的 train N={len(tr)}\n",
         "| 模型 | " + " | ".join(f"{g}（{int(m.sum())}）" for g, m in groups.items())
         + " | 年齡配對（全部） | 年齡配對（壓迫性） | Spearman(分數,年齡)陰性 |",
         "|---" * (len(groups) + 4) + "|"]
    for name, p in preds.items():
        p = np.asarray(p, float)
        cells = []
        for g, m in groups.items():
            sel = (m | neg).values & ~np.isnan(p)
            cells.append(f"{auroc(m.values[sel].astype(float), p[sel]):.3f}")
        for g in ("全部骨折", "椎體壓迫性"):
            sel = (groups[g] | neg).values & ~np.isnan(p) & ~np.isnan(agev)
            cells.append(f"{matched_auc(groups[g].values[sel].astype(int), p[sel], agev[sel], 2.0):.3f}")
        sn = neg.values & ~np.isnan(p) & ~np.isnan(agev)
        cells.append(f"{spearmanr(p[sn], agev[sn])[0]:+.3f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    L += ["", f"## Paired bootstrap（val，{a.n_boot} 次，重抽 study；括號為 95% CI；p = P(差≤0)）\n",
          "| 模型 | 對照 | 評估族群 | ΔAUROC | 95% CI | p | Δ年齡配對 | 95% CI | p |", "|---|---|---|---|---|---|---|---|---|"]
    cmp_rows = [n for n in preds if "E5_" in n]
    vc = "valCV:stackB" if a.bone_oof else "valCV:stack"
    for mname in cmp_rows:
        refs = ["E0(probe)", best] + ([vc] if mname.startswith("valCV") else []) \
            + (["bone_ens(F1)"] if a.bone_ens and mname.startswith(("avg", "E5_")) else [])
        for ref in refs:
            for g in ("全部骨折", "椎體壓迫性"):
                sel = (groups[g] | neg).values & ~np.isnan(agev)
                r = boot_diff(groups[g].values[sel].astype(int), np.asarray(preds[mname], float)[sel],
                              np.asarray(preds[ref], float)[sel], agev[sel], 2.0, a.n_boot)
                d, lo, hi, p = r["auc"]; dm, lom, him, pm = r["matched"]
                L.append(f"| {mname} | {ref} | {g} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {dm:+.3f} | "
                         f"[{lom:+.3f}, {him:+.3f}] | {pm:.3f} |")
    Path(a.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
