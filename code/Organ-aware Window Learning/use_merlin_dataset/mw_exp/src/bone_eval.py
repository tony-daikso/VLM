"""Experiment B report: bone-unit fracture models vs E0 (lumbar_spine) and vs the current best OOF stack. VAL only.

Inputs (all train/val only; test rows are filtered out on load and asserted absent where possible):
  agg/*.npz            val logits of bone aggregation models (bone_probe.py --stage agg)
  std/E0.npz           lumbar_spine E0 probe (val rows only are read)
  oof_E3late.npz       E3_late OOF logits (train OOF + val 5-fold mean)
  oof_bone_*.npz       bone OOF logits (bone_probe.py --stage oof)
  fracture_features_*  geometry/HU features
Stacks (fit on TRAIN with OOF logits, scored on VAL; HistGB recipe of fracture_eval_oof.py):
  stack_ref    = GB(features + E3_late OOF)             (current best, .788)
  stack+bone   = GB(features + E3_late OOF + bone OOF)
  lr2          = LR(E3_late OOF, bone OOF)
Metrics on the common val set: AUROC, paired bootstrap ΔAUROC/Δage-matched vs E0 and vs stack_ref,
age-matched AUROC (±2 y), Spearman(score, age | negatives), per-site AUROC (site positives vs all negatives).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval import load_labels  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

R = Path.home() / "Desktop/Tony/VLM/OWL"
FR = R / "mw_exp/outputs_fracture"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def to_md(df):
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    L = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    L += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones", default=str(FR / "bones"))
    ap.add_argument("--agg", nargs="*", default=None, help="agg model names (default: all in agg/)")
    ap.add_argument("--bone-oof", nargs="*", default=[], help="oof_bone_*.npz paths for stacking")
    ap.add_argument("--extra-val", nargs="*", default=[], help="name=path.npz with ids/logit (val) to report too")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.bones)

    spl = pd.read_csv(R / "data/split.csv").set_index("study_id")["split"]
    spl = spl[spl.isin(["train", "val"])]
    lab = load_labels(R / "data/labels_disease14_all.jsonl")
    lab = lab[lab.index.isin(spl.index) & ~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    feat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(FR / "fracture_features_*.csv")))]).drop_duplicates("id", keep="last").set_index("id")
    allf = list(feat.columns)
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age")).join(feat)

    z = np.load(FR / "oof_E3late.npz", allow_pickle=True)
    assert not np.isin(z["split"], ["test"]).any()
    df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("e3_oof"))
    z = np.load(R / "mw_exp/outputs_final/results/std/E0.npz", allow_pickle=True)
    vm = z["split"] == "val"                                       # only val rows are read
    df = df.join(pd.Series(logit(z["prob"].mean(0)[vm, list(z["findings"]).index("fracture")]),
                           index=np.array(z["ids"])[vm]).rename("E0"))
    z = np.load(R / "mw_exp/outputs_final/results/std/E3_late.npz", allow_pickle=True)
    vm = z["split"] == "val"
    df = df.join(pd.Series(logit(z["prob"].mean(0)[vm, list(z["findings"]).index("fracture")]),
                           index=np.array(z["ids"])[vm]).rename("E3_late"))
    for p in a.bone_oof:
        z = np.load(p, allow_pickle=True)
        assert set(z["split"]) <= {"train", "val"}
        df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("oof:" + Path(p).stem.replace("oof_bone_", "")))
    names = a.agg or sorted(Path(p).stem for p in glob.glob(str(B / "agg/*.npz")))
    for n in names:
        z = np.load(B / "agg" / f"{n}.npz")
        df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("agg:" + n))
    for e in a.extra_val:
        n, p = e.split("=")
        z = np.load(p, allow_pickle=True)
        df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename(n))

    tr, va = df[df.split == "train"], df[df.split == "val"].copy()
    need = ["E0", "e3_oof"] + [c for c in df.columns if c.startswith(("agg:", "oof:"))]
    va = va.dropna(subset=need + ["age"])
    preds = {"E0（腰椎單元，參考）": va["E0"].values, "E3_late（原 probe）": va["E3_late"].values,
             "E3_late OOF": va["e3_oof"].values}

    def fit_gb(cols, rows):
        rows = rows.dropna(subset=[c for c in cols if c in ("e3_oof",) or c.startswith("oof:")])
        return gb().fit(rows[cols].values, rows["y"].values).predict_proba(va[cols].values)[:, 1]

    def fit_lr(cols, rows):
        rows = rows.dropna(subset=cols)
        sc = StandardScaler().fit(rows[cols].values)
        return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(rows[cols].values), rows["y"].values) \
            .decision_function(sc.transform(va[cols].values))
    preds["stack_ref：GB(幾何特徵 + E3_late OOF)"] = fit_gb(allf + ["e3_oof"], tr)
    for c in [c for c in df.columns if c.startswith("oof:")]:
        n = c[4:]
        preds[f"骨骼 OOF：{n}"] = va[c].values
        preds[f"GB(幾何特徵 + E3_late OOF + 骨骼 OOF {n})"] = fit_gb(allf + ["e3_oof", c], tr)
        preds[f"LR(E3_late OOF + 骨骼 OOF {n})"] = fit_lr(["e3_oof", c], tr)
        preds[f"GB(幾何特徵 + 骨骼 OOF {n})"] = fit_gb(allf + [c], tr)
    for c in [c for c in df.columns if c.startswith("agg:")]:
        preds[f"骨骼（val 選 HP）：{c[4:]}"] = va[c].values
    for e in a.extra_val:
        preds[e.split("=")[0]] = va[e.split("=")[0]].values

    y = va["y"].values.astype(int)
    agev = va["age"].values
    neg = y == 0
    groups = {
        "全部骨折": y == 1,
        "椎體": (y == 1) & va["sites"].apply(lambda s: "vertebra" in s).values,
        "椎體壓迫性": (y == 1) & va["compression"].values.astype(bool),
        "肋骨": (y == 1) & va["sites"].apply(lambda s: "rib" in s).values,
        "骨盆/薦椎/髖": (y == 1) & va["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))).values,
        "急性(含混合)": (y == 1) & va["acute"].isin(["yes", "mixed"]).values,
        "陳舊": (y == 1) & (va["acute"] == "no").values,
    }
    ref = preds["stack_ref：GB(幾何特徵 + E3_late OOF)"]
    rows = []
    for name, p in preds.items():
        p = np.asarray(p, float)
        r = {"模型": name, "AUROC": auroc(y.astype(float), p), "年齡配對": matched_auc(y, p, agev, 2.0),
             "Spearman(分數,年齡|陰性)": spearmanr(p[neg], agev[neg])[0]}
        for k, (base, tag) in enumerate(((preds["E0（腰椎單元，參考）"], "vs E0"), (ref, "vs stack_ref"))):
            if name.startswith("E0（") and k == 0 or name.startswith("stack_ref") and k == 1:
                continue
            b = boot_diff(y, p, np.asarray(base, float), agev, 2.0, a.n_boot)
            r[f"Δ{tag}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
            r[f"Δ年齡配對 {tag}"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
        for g, m in list(groups.items())[1:]:
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(a.out)
    res.to_csv(out.with_suffix(".csv"), index=False)
    hdr = f"# 實驗 B：骨骼單元（val 共同集 N={len(va)}，骨折 {int(y.sum())}，陰性 {int(neg.sum())}；train N={len(tr)}）\n"
    hdr += "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + "\n\n"
    out.write_text(hdr + to_md(res) + "\n")
    print(hdr + to_md(res))


if __name__ == "__main__":
    main()
