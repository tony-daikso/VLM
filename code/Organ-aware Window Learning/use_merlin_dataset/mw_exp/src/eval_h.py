"""Experiment H report (VAL only; test never read): finetuned all-bone site-MIL (finetune_bones.py pred_val.npz)
vs bone_ens (.826) and E0 (lumbar unit, .777) on the common val set.

Columns: val AUROC, paired bootstrap ΔAUROC / Δage-matched (±2 y) vs bone_ens and vs E0, age-matched AUROC,
Spearman(score, age | negatives), per-site AUROC (site positives vs all negatives).
Complementarity: FIXED equal-weight mean of z-scored logits (z-score over the val rows, label-free; no weight tuning).
usage: python eval_h.py --pred H1=/path/pred_val.npz [--pred H0=...] [--pred-inner ...] --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from final_stack import FR, R, SITES, load_labels_trval, logit, split_trval, to_md  # noqa: E402
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=pred_val.npz (ids, split, logit)")
    ap.add_argument("--bone-ens", default=str(FR / "bones/final_F1/groups/bone_ens.npz"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spl = split_trval()
    lab = load_labels_trval(set(spl.index))
    lab = lab[~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age"))
    df = df[df.split == "val"]
    z = np.load(R / "mw_exp/outputs_final/results/std/E0.npz", allow_pickle=True)
    vm = z["split"] == "val"
    df = df.join(pd.Series(logit(z["prob"].mean(0)[vm, list(z["findings"]).index("fracture")]),
                           index=np.array(z["ids"])[vm]).rename("E0"))
    z = np.load(a.bone_ens, allow_pickle=True)
    vm = z["split"] == "val"
    df = df.join(pd.Series(z["logit"][vm], index=np.array(z["ids"])[vm].astype(str)).rename("bone_ens"))
    names = []
    for spec in a.pred:
        n, p = spec.split("=", 1)
        z = np.load(p, allow_pickle=True)
        assert set(np.unique(z["split"])) == {"val"}
        df = df.join(pd.Series(z["logit"], index=np.array(z["ids"]).astype(str)).rename(n)); names.append(n)
    va = df.dropna(subset=["E0", "bone_ens", "age"] + names).copy()
    zs = lambda v: (v - v.mean()) / v.std()
    for n in names:
        va[f"{n}+bone_ens"] = (zs(va[n]) + zs(va["bone_ens"])) / 2
    y = va["y"].values.astype(int); agev = va["age"].values.astype(float); neg = y == 0
    groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in SITES.items()}
    preds = [("E0 腰椎單元（參考 .777）", "E0"), ("bone_ens 凍結 10-seed 集成（參考 .826）", "bone_ens")]
    preds += [(f"實驗 H {n}", n) for n in names] + [(f"等權 z 平均：{n} + bone_ens", f"{n}+bone_ens") for n in names]
    rows = []
    for label, c in preds:
        p = va[c].values.astype(float)
        r = {"模型": label, "val AUROC": auroc(y.astype(float), p), "年齡配對(±2y)": matched_auc(y, p, agev, 2.0),
             "Spearman(分數,年齡|陰性)": spearmanr(p[neg], agev[neg])[0]}
        for base, tag in [("bone_ens", "vs bone_ens"), ("E0", "vs E0")]:
            if c == base:
                r[f"Δ{tag}"] = r[f"Δ年齡配對 {tag}"] = "—"; continue
            b = boot_diff(y, p, va[base].values.astype(float), agev, 2.0, a.n_boot)
            r[f"Δ{tag}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
            r[f"Δ年齡配對 {tag}"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
        for g, m in groups.items():
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "report_h.csv", index=False)
    hdr = f"# 實驗 H：全骨骼 finetune（val 共同集 N={len(va)}，骨折 {int(y.sum())}，陰性 {int(neg.sum())}）\n"
    hdr += "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + "\n\n"
    hdr += "互補性：固定等權（各自在 val 上做無標籤 z-score 後平均），未在 val 上調權重。\n\n"
    (out / "report_h.md").write_text(hdr + to_md(res) + "\n")
    print(hdr + to_md(res))
    corr = {n: float(np.corrcoef(va[n], va["bone_ens"])[0, 1]) for n in names}
    (out / "summary.json").write_text(json.dumps({"n_val": len(va), "pearson_vs_bone_ens": corr,
                                                  "rows": res.to_dict(orient="records")}, indent=1, ensure_ascii=False))
    print("Pearson(logit, bone_ens):", corr)


if __name__ == "__main__":
    main()
