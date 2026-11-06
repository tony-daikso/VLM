"""Experiment E report: label-refinement variants of the bone-unit fracture model. VAL only (test never loaded).

References (same common val set as bone_eval.py, N=4,946):
  bone_ref  = Exp B bone OOF site_max__mean__all3                      (.822)
  stack_ref = GB(geometry features + E3_late OOF + bone_ref OOF)        (.823)
For every variant v (labelE/oof_bone_E_<v>.npz): bone OOF alone (vs bone_ref) and GB(features + E3_late OOF + v OOF)
(vs stack_ref). Val target is always the hard y. CV column = 5-fold OOF AUROC on TRAIN (selection metric).
Acute / old heads: also LR(head_acute, head_old) fit on train OOF and the GB stack with both.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from bone_eval import FR, R, gb, logit, to_md  # noqa: E402
from fracture_eval import load_labels  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

NAMES = {"ref": "參考（硬標籤，本程式 CPU 重跑）", "excl_amb": "E1a 排除 probably_yes/no", "soft_cat": "E1b 軟標籤 py=0.7/pn=0.2",
         "soft_p": "E1c 軟標籤 p/100", "soft_p_unc": "E1d 軟標籤 p/100 + 加入 unclear", "soft_p_site": "E2 軟標籤 + 部位輔助頭軟標籤",
         "acute_w2": "E3a 急性加權 ×2", "acute_w4": "E3b 急性加權 ×4", "head_acute": "E3c 急性專用頭", "head_old": "E3c 陳舊專用頭",
         "nm_drop": "E4a 移除 not_mentioned 陰性", "nm_w03": "E4b not_mentioned 權重 0.3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones", default=str(FR / "bones"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ens", action="store_true", help="use the 3-seed mean logit of each variant (seed 0,1,2)")
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
    vm = z["split"] == "val"                                       # only val rows are read (common-set definition)
    df = df.join(pd.Series(logit(z["prob"].mean(0)[vm, list(z["findings"]).index("fracture")]),
                           index=np.array(z["ids"])[vm]).rename("E0"))
    cv = {}
    srcs = [("bone_ref", B / "oof_bone_site_max__mean__all3.npz")]
    seed_info = {}
    for v in NAMES:
        fs = [B / f"labelE/oof_bone_E_{v}.npz"] + [B / f"labelE/oof_bone_E_{v}_s{k}.npz" for k in (1, 2)]
        fs = [f for f in fs if f.exists()]
        if not fs:
            continue
        if not a.ens:
            fs = fs[:1]
        elif len(fs) < 3:
            continue
        srcs.append((v, fs))
    for n, fs in srcs:
        fs = fs if isinstance(fs, list) else [fs]
        Ls, cvs, vals = [], [], []
        for p in fs:
            z = np.load(p, allow_pickle=True)
            assert set(z["split"]) <= {"train", "val"}
            Ls.append(z["logit"]); trm, vam = z["split"] == "train", z["split"] == "val"
            cvs.append(auroc(z["y"][trm], z["logit"][trm])); vals.append(auroc(z["y"][vam], z["logit"][vam]))
        df = df.join(pd.Series(np.mean(Ls, 0), index=list(z["ids"])).rename("oof:" + n))
        cv[n] = float(np.mean(cvs))
        seed_info[n] = (cvs, vals)
    oofc = [c for c in df.columns if c.startswith("oof:")]
    tr, va = df[df.split == "train"], df[df.split == "val"].copy()
    va = va.dropna(subset=["E0", "e3_oof", "age"] + oofc)

    def fit_gb(cols):
        rows = tr.dropna(subset=[c for c in cols if c == "e3_oof" or c.startswith("oof:")])
        return gb().fit(rows[cols].values, rows["y"].values).predict_proba(va[cols].values)[:, 1]

    def fit_lr(cols):
        rows = tr.dropna(subset=cols)
        sc = StandardScaler().fit(rows[cols].values)
        return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(rows[cols].values), rows["y"].values) \
            .decision_function(sc.transform(va[cols].values))

    bone_ref = va["oof:bone_ref"].values
    stack_ref = fit_gb(allf + ["e3_oof", "oof:bone_ref"])
    preds = [("實驗 B 骨骼 OOF site_max（.822 參考）", bone_ref, cv["bone_ref"], "bone"),
             ("實驗 B stack：GB(幾何 + E3_late OOF + 骨骼 OOF)（.823 參考）", stack_ref, np.nan, "stack")]
    order = [v for v in NAMES if f"oof:{v}" in va.columns]
    for v in order:
        preds.append((f"{NAMES[v]}：骨骼 OOF", va[f"oof:{v}"].values, cv[v], "bone"))
        preds.append((f"{NAMES[v]}：GB stack", fit_gb(allf + ["e3_oof", f"oof:{v}"]), np.nan, "stack"))
    if {"oof:head_acute", "oof:head_old"} <= set(va.columns):
        preds.append(("E3c LR(急性頭 + 陳舊頭)", fit_lr(["oof:head_acute", "oof:head_old"]), np.nan, "bone"))
        preds.append(("E3c LR(參考骨骼 + 急性頭 + 陳舊頭)", fit_lr(["oof:bone_ref", "oof:head_acute", "oof:head_old"]), np.nan, "bone"))
        preds.append(("E3c GB(幾何 + E3_late OOF + 急性頭 + 陳舊頭)", fit_gb(allf + ["e3_oof", "oof:head_acute", "oof:head_old"]), np.nan, "stack"))

    y = va["y"].values.astype(int)
    agev = va["age"].values
    neg = y == 0
    groups = {
        "椎體壓迫性": (y == 1) & va["compression"].values.astype(bool),
        "肋骨": (y == 1) & va["sites"].apply(lambda s: "rib" in s).values,
        "骨盆/薦椎/髖": (y == 1) & va["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))).values,
        "急性(含混合)": (y == 1) & va["acute"].isin(["yes", "mixed"]).values,
        "陳舊": (y == 1) & (va["acute"] == "no").values,
    }
    rows = []
    for name, p, c, kind in preds:
        p = np.asarray(p, float)
        r = {"模型": name, "train 5-fold OOF AUROC": c, "val AUROC": auroc(y.astype(float), p),
             "年齡配對": matched_auc(y, p, agev, 2.0), "Spearman(分數,年齡|陰性)": spearmanr(p[neg], agev[neg])[0]}
        for base, tag in ((bone_ref, "vs .822"), (stack_ref, "vs .823")):
            if np.array_equal(p, base):
                r[f"Δ{tag}"] = r[f"Δ年齡配對 {tag}"] = "—"
                continue
            b = boot_diff(y, p, np.asarray(base, float), agev, 2.0, a.n_boot)
            r[f"Δ{tag}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
            r[f"Δ年齡配對 {tag}"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
        for g, m in groups.items():
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(a.out)
    res.to_csv(out.with_suffix(".csv"), index=False)
    hdr = f"# 實驗 E：標籤改良（val 共同集 N={len(va)}，骨折 {int(y.sum())}，陰性 {int(neg.sum())}；train N={len(tr)}）\n"
    hdr += "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + "\n\n"
    if a.ens:
        hdr += "（3 個 seed 的 logit 平均；train OOF AUROC 為 3 個 seed 的平均。）\n\n"
    sd = ["| 變體 | train OOF AUROC 各 seed | 平均 ± SD | val AUROC 各 seed（未過濾共同集） | 平均 ± SD |", "|---|---|---|---|---|"]
    for n, (c, v) in seed_info.items():
        sd.append(f"| {NAMES.get(n, n)} | {', '.join(f'{x:.4f}' for x in c)} | {np.mean(c):.4f} ± {np.std(c):.4f} | "
                  f"{', '.join(f'{x:.4f}' for x in v)} | {np.mean(v):.4f} ± {np.std(v):.4f} |")
    out.write_text(hdr + to_md(res) + "\n\n## 各 seed\n\n" + "\n".join(sd) + "\n")
    print(hdr + to_md(res))


if __name__ == "__main__":
    main()
