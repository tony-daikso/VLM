"""Exp I evaluation. Each --g name=glob[,glob] is seed-averaged (OOF member format). Train rows = 5-fold OOF (selection),
val rows = mean of fold models (report only; printed only with --split val). Test never read (meta.csv is train/val).
Per group: AUROC, age-matched AUROC (±2 y), Spearman(score, age | negatives), subgroup AUROCs (subgroup positives vs all
negatives), and on val paired-bootstrap ΔAUROC / Δage-matched vs bone_ens (.826) and vs E3_late 10-seed OOF (.831).
usage: python bone_expI_eval.py --split train --g ref3='.../oof_bone_E_ref__E3_late_s[0-2].npz' --g lv1='.../oof_bone_E_lv1_s*.npz'
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
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
B = MW / "outputs_fracture/bones"
REF = {"bone_ens": str(B / "final_F1/groups/bone_ens.npz"), "E3_late_OOF": str(B / "rerun/std/oof/oof_bone_E_ref__E3_late_s*.npz")}


FOLD = None


def fold_of(meta):
    """train fold id of modes_oof / bone_expI (StratifiedKFold(5, shuffle, rs=42) over train rows in bone_matrix order)."""
    from sklearn.model_selection import StratifiedKFold
    tr = np.where(meta.split.values == "train")[0]
    f = pd.Series(-1, index=meta.index)
    for k, (_, h) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, meta.y.values[tr])):
        f.iloc[tr[h]] = k
    return f


def load_group(spec, key="logit", foldz=False):
    """seed mean; foldz: train OOF logits z-scored within each CV fold first (removes cross-fold calibration offsets,
    which otherwise dominate pooled OOF AUROC differences between variants; val rows untouched)."""
    fs = sorted({f for g in spec.split(",") for f in glob.glob(g)})
    assert fs, spec
    L = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}
        s = pd.Series(np.asarray(z[key], float), index=[str(i) for i in z["ids"]])
        if foldz:
            fo = FOLD.reindex(s.index)
            for k in range(5):
                mk = (fo == k).values
                s[mk] = (s[mk] - s[mk].mean()) / s[mk].std()
        L.append(s)
    return pd.concat(L, axis=1).mean(1), len(fs)


def subgroups(m):
    pos = m.y == 1
    ro = pos & m.site_rib & ~m.site_vert & ~m.site_pelv & ~m.comp
    po = pos & m.site_pelv & ~m.site_vert & ~m.site_rib & ~m.comp
    return {"椎體壓迫性": pos & m.comp, "輕度壓迫": pos & m.mild, "肋骨": pos & m.site_rib, "肋骨單獨": ro,
            "骨盆/薦椎/髖": pos & m.site_pelv, "骨盆單獨": po, "急性": pos & (m.grp == "acute"), "陳舊": pos & (m.grp == "old"),
            "臨床顯著": pos & m.cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", action="append", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--raw", action="store_true", help="train: pooled raw OOF logits (default: fold-z-scored)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = pd.read_csv(B / "expI/meta.csv", index_col=0, keep_default_na=False, na_values=[""])
    assert set(meta.split) <= {"train", "val"}
    for c in ["comp", "mild", "site_vert", "site_rib", "site_pelv", "site_hip", "cs"]:
        meta[c] = meta[c].astype(str).eq("True")
    global FOLD
    FOLD = fold_of(meta)
    m = meta[meta.split == a.split].copy()
    specs = ([] if a.no_ref else list(REF.items())) + [tuple(g.split("=", 1)) for g in a.g]
    nseeds = {}
    for n, sp in specs:
        m[n], nseeds[n] = load_group(sp, foldz=(a.split == "train" and not a.raw))
    m = m.dropna(subset=[n for n, _ in specs] + ["y"])
    y = m.y.values.astype(int); age = m.age.values.astype(float); neg = y == 0; okage = ~np.isnan(age)
    G = subgroups(m)
    rows = []
    for n, _ in specs:
        p = m[n].values
        r = {"群組": n, "seeds": nseeds[n], "AUROC": auroc(y.astype(float), p),
             "年齡配對": matched_auc(y[okage], p[okage], age[okage], 2.0),
             "Spearman(陰性)": spearmanr(p[neg & okage], age[neg & okage])[0]}
        if a.split == "val" and not a.no_ref:
            for rn in REF:
                if rn == n:
                    r[f"Δ vs {rn}"] = r[f"Δ年齡配對 vs {rn}"] = "—"; continue
                b = boot_diff(y[okage], p[okage], m[rn].values[okage], age[okage], 2.0, a.n_boot)
                da = auroc(y.astype(float), p) - auroc(y.astype(float), m[rn].values)
                r[f"Δ vs {rn}"] = f"{da:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
                r[f"Δ年齡配對 vs {rn}"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
        for g, mk in G.items():
            sel = mk.values | neg
            r[g] = auroc(mk.values[sel].astype(float), p[sel])
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(f"split={a.split} N={len(m)} pos={int(y.sum())} neg={int(neg.sum())} | " + ", ".join(f"{g} {int(v.sum())}" for g, v in G.items()))
    print(df.round(4).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
