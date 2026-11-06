"""Experiment F: final fracture stack from a list of OOF members. Fit on TRAIN (OOF logits), report on VAL. Test never read.

Each member is an npz in the oof format (ids, split in {train, val}, logit): train rows = 5-fold OOF logit,
val rows = mean of the fold models (bone_probe.py --stage oof, bone_label_exp.py, oof_fracture.py, bone_ens.py).
A member given as several files / globs is averaged (e.g. 10 seeds of one config).
Labels: split.csv is filtered to train/val FIRST; only jsonl lines whose study_id is in that set are parsed.

Stack inputs: [members..., E3_late OOF (unless --no-e3), geometry/HU features (unless --no-feat)].
Stacker: gb (HistGB recipe of bone_eval.py) or lr (standardized, median-imputed + missing indicators); --stacker auto
picks the one with the higher 5-fold CV AUROC on TRAIN (stacker CV on the OOF rows; val is never used to choose).
References on the same common val set: E0 (lumbar unit, .777) and the Exp B stack GB(features + E3_late OOF +
bone OOF site_max__mean__all3) (.823), recomputed here.
Report: val AUROC + paired bootstrap ΔAUROC / Δage-matched vs both refs, age-matched AUROC (±2 y),
Spearman(score, age | negatives), per-site AUROC (site positives vs all negatives).
Outputs (--out DIR): report.md / report.csv, scores.npz (ids, split, y, stack [train = stacker 5-fold CV OOF,
val = fit on all train], one array per member), config.json (members, files, train CV, chosen stacker).

usage:
  python final_stack.py --member bone=../outputs_fracture/bones/final_F1/groups/bone_ens.npz --out DIR
  # add the finetune logit later (npz with ids/split/logit, train OOF + val):
  python final_stack.py --member bone=.../final_F1/groups/bone_ens.npz --member ft=/path/oof_ft.npz --out DIR2
  # several files per member are averaged:  --member ref10='.../members/oof_bone_E_ref*.npz'
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

R = Path.home() / "Desktop/Tony/VLM/OWL"
FR = R / "mw_exp/outputs_fracture"
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}
SUMMARY = ["n_levels", "min_ant_post_ratio", "min_height_nb_ratio", "max_height_loss", "min_vol_nb_ratio",
           "L1_hu", "T12_L2_hu", "min_body_hu_p10"]
SITES = {
    "椎體壓迫性": lambda d: d["compression"],
    "肋骨": lambda d: d["sites"].apply(lambda s: "rib" in s),
    "骨盆/薦椎/髖": lambda d: d["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))),
    "急性(含混合)": lambda d: d["acute"].isin(["yes", "mixed"]),
    "陳舊": lambda d: d["acute"] == "no",
}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def split_trval():
    spl = pd.read_csv(R / "data/split.csv").set_index("study_id")["split"]
    return spl[spl.isin(["train", "val"])]                                   # test ids dropped before anything else


def load_labels_trval(keep):
    """Fracture labels for train/val ids only: the id is matched on the raw line before the line is parsed."""
    rx, rows = re.compile(r'"study_id"\s*:\s*"([^"]+)"'), {}
    for line in open(R / "data/labels_disease14_all.jsonl"):
        m = rx.search(line)
        if not m or m.group(1) not in keep:
            continue
        r = json.loads(line)
        a = r["answer"]["fracture"]
        d = a.get("detail", {}) or {}
        cat = a["category"]
        rows[r["study_id"]] = {"y": 1.0 if cat in POS else 0.0 if cat in NEG else np.nan,
                               "compression": d.get("vertebral_compression") == "yes",
                               "sites": d.get("sites", []) or [], "acute": d.get("acute", "not_stated")}
    return pd.DataFrame.from_dict(rows, orient="index")


def load_member(spec):
    files = sorted({f for g in spec.split(",") for f in glob.glob(g)})
    assert files, f"no files for {spec}"
    Ls, ids0 = [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}, f"{f}: non train/val rows"
        s = pd.Series(np.asarray(z["logit"], float), index=[str(i) for i in z["ids"]])
        Ls.append(s if ids0 is None else s.reindex(ids0)); ids0 = ids0 if ids0 is not None else s.index
    return pd.concat(Ls, axis=1, sort=False).mean(1), files


def lr_design(Xtr, Xva, nfeat_cols):
    """median impute + missing indicators for feature columns; standardize everything (train stats)."""
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for c in nfeat_cols:
        med = Xtr[c].median()
        if Xtr[c].isna().any() or Xva[c].isna().any():
            Xtr[c + "_na"], Xva[c + "_na"] = Xtr[c].isna().astype(float), Xva[c].isna().astype(float)
        Xtr[c], Xva[c] = Xtr[c].fillna(med), Xva[c].fillna(med)
    sc = StandardScaler().fit(Xtr.values)
    return sc.transform(Xtr.values), sc.transform(Xva.values)


def fit_score(kind, tr, va, cols, feat_cols):
    if kind == "gb":
        return gb().fit(tr[cols].values, tr["y"].values).predict_proba(va[cols].values)[:, 1]
    lr_feat = [c for c in feat_cols if c in SUMMARY]                         # LR: 8 summary features only
    lcols = [c for c in cols if c not in feat_cols] + lr_feat
    A, B = lr_design(tr[lcols], va[lcols], lr_feat)
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(A, tr["y"].values).decision_function(B)


def cv_oof(kind, tr, cols, feat_cols, folds=5):
    o = np.zeros(len(tr))
    for fi, (a, b) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=0).split(tr, tr["y"])):
        o[b] = fit_score(kind, tr.iloc[a], tr.iloc[b], cols, feat_cols)
        if kind == "gb":
            o[b] = logit(o[b])
    return o


def to_md(df):
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    L = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(L + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="name=npz[,npz|glob...] (files averaged)")
    ap.add_argument("--no-e3", action="store_true"); ap.add_argument("--no-feat", action="store_true")
    ap.add_argument("--stacker", default="auto", choices=["auto", "gb", "lr"])
    ap.add_argument("--bone-ref", default=str(FR / "bones/oof_bone_site_max__mean__all3.npz"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--extra-ref", action="append", default=[],
                    help="name=npz[,glob]: extra reference (OOF format) shown as a row and as a ΔAUROC column (e.g. bone_ens .826)")
    ap.add_argument("--feat-extra", action="append", default=[],
                    help="csv with an id column + extra stack features (e.g. geom15_features.csv); added to the GB "
                         "features and to the LR summary features")
    ap.add_argument("--title", default="實驗 F：最終整合")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    spl = split_trval()
    lab = load_labels_trval(set(spl.index))
    lab = lab[~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    feat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(FR / "fracture_features_*.csv")))]) \
        .drop_duplicates("id", keep="last").set_index("id")
    feat = feat[feat.index.isin(spl.index)]
    base_feat = list(feat.columns)                                            # refs keep the original features
    for fx in a.feat_extra:
        ex = pd.read_csv(fx).drop_duplicates("id").set_index("id")
        ex = ex[ex.index.isin(spl.index)]
        feat = feat.join(ex, how="left")
        SUMMARY.extend([c for c in ex.columns if c not in SUMMARY])
    feat_cols = [] if a.no_feat else list(feat.columns)
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age"))
    if feat_cols:
        df = df.join(feat)
    else:
        df = df.join(feat, rsuffix="_unused")                                  # refs still need the features
    z = np.load(FR / "oof_E3late.npz", allow_pickle=True)
    assert not np.isin(z["split"], ["test"]).any()
    df = df.join(pd.Series(z["logit"], index=list(z["ids"])).rename("e3_oof"))
    z = np.load(R / "mw_exp/outputs_final/results/std/E0.npz", allow_pickle=True)
    vm = z["split"] == "val"                                                  # only val rows are read
    df = df.join(pd.Series(logit(z["prob"].mean(0)[vm, list(z["findings"]).index("fracture")]),
                           index=np.array(z["ids"])[vm]).rename("E0"))
    df = df.join(load_member(a.bone_ref)[0].rename("bone_ref"))
    xrefs = []
    for spec in a.extra_ref:
        n, p = spec.split("=", 1)
        df = df.join(load_member(p)[0].rename("x:" + n)); xrefs.append(n)
    members, files, mcols = {}, {}, []
    for spec in a.member:
        n, p = spec.split("=", 1)
        s, fs = load_member(p)
        df = df.join(s.rename("m:" + n)); files[n] = fs; mcols.append("m:" + n)
    cols = mcols + ([] if a.no_e3 else ["e3_oof"]) + feat_cols
    allf = base_feat
    tr = df[df.split == "train"].dropna(subset=mcols + ["e3_oof"]).copy()
    va = df[df.split == "val"].dropna(subset=mcols + ["E0", "e3_oof", "age", "bone_ref"] + ["x:" + n for n in xrefs]).copy()

    # stacker choice on TRAIN CV only
    cv = {}
    kinds = ["gb", "lr"] if a.stacker == "auto" else [a.stacker]
    oofs = {}
    for k in kinds:
        oofs[k] = cv_oof(k, tr, cols, feat_cols)
        cv[k] = auroc(tr["y"].values, oofs[k])
    kind = max(cv, key=cv.get)
    print("stacker train CV AUROC:", {k: round(v, 4) for k, v in cv.items()}, "->", kind, flush=True)
    stack = fit_score(kind, tr, va, cols, feat_cols)
    trb = df[df.split == "train"].dropna(subset=["e3_oof", "bone_ref"])
    ref_b = gb().fit(trb[allf + ["e3_oof", "bone_ref"]].values, trb["y"].values).predict_proba(va[allf + ["e3_oof", "bone_ref"]].values)[:, 1]

    y = va["y"].values.astype(int); agev = va["age"].values.astype(float); neg = y == 0
    groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in SITES.items()}
    preds = [("E0 腰椎單元（參考 .777）", va["E0"].values, np.nan),
             ("實驗 B 骨骼 OOF site_max（單 seed，.822）", va["bone_ref"].values, auroc(tr["y"].values, tr["bone_ref"].values) if tr["bone_ref"].notna().all() else np.nan),
             ("實驗 B stack：GB(幾何 + E3_late OOF + 骨骼 OOF)（參考 .823）", ref_b, np.nan)]
    for n in xrefs:
        preds.append((f"參考 {n}（OOF logit）", va["x:" + n].values, auroc(tr["y"].values, tr["x:" + n].values) if tr["x:" + n].notna().all() else np.nan))
    for c in mcols:
        preds.append((f"成員 {c[2:]}（OOF logit）", va[c].values, auroc(tr["y"].values, tr[c].values)))
    sname = f"最終 stack：{kind.upper()}({' + '.join([c[2:] for c in mcols] + ([] if a.no_e3 else ['E3_late OOF']) + ([] if a.no_feat else ['幾何/HU']))})"
    for k in kinds:                                                         # non-chosen stacker shown for reference only
        if k != kind:
            preds.append((sname.replace(kind.upper() + "(", k.upper() + "(").replace("最終 stack", "（未選）stack"),
                          fit_score(k, tr, va, cols, feat_cols), cv[k]))
    preds.append((sname, stack, cv[kind]))
    rows = []
    for name, p, c in preds:
        p = np.asarray(p, float)
        r = {"模型": name, "train CV AUROC": c, "val AUROC": auroc(y.astype(float), p), "年齡配對(±2y)": matched_auc(y, p, agev, 2.0),
             "Spearman(分數,年齡|陰性)": spearmanr(p[neg], agev[neg])[0]}
        for base, tag in [(va["E0"].values, "vs E0"), (ref_b, "vs .823")] + [(va["x:" + n].values, f"vs {n}") for n in xrefs]:
            if np.array_equal(p, base):
                r[f"Δ{tag}"] = r[f"Δ年齡配對 {tag}"] = "—"; continue
            b = boot_diff(y, p, np.asarray(base, float), agev, 2.0, a.n_boot)
            r[f"Δ{tag}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
            r[f"Δ年齡配對 {tag}"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
        for g, m in groups.items():
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "report.csv", index=False)
    hdr = f"# {a.title}（val 共同集 N={len(va)}，骨折 {int(y.sum())}，陰性 {int(neg.sum())}；stack train N={len(tr)}）\n"
    hdr += "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + "\n\n"
    hdr += f"stacker 以 train 5-fold CV 選擇：{', '.join(f'{k.upper()} {v:.4f}' for k, v in cv.items())} → {kind.upper()}（val 未參與任何選擇）\n\n"
    (out / "report.md").write_text(hdr + to_md(res) + "\n")
    print(hdr + to_md(res))

    ids = np.r_[tr.index.values, va.index.values].astype(str)
    sv = {"ids": ids, "split": np.r_[["train"] * len(tr), ["val"] * len(va)], "y": np.r_[tr["y"].values, va["y"].values],
          "stack": np.r_[oofs[kind], logit(stack) if kind == "gb" else stack], "stack_kind": kind}
    for c in mcols + ["e3_oof"]:
        sv["m_" + c.replace("m:", "")] = np.r_[tr[c].values, va[c].values]
    np.savez(out / "scores.npz", **sv)
    (out / "config.json").write_text(json.dumps({"members": files, "feat_extra": a.feat_extra, "e3": not a.no_e3, "feat": not a.no_feat, "stacker_cv": cv,
                                                 "stacker": kind, "n_train": len(tr), "n_val": len(va),
                                                 "val_auc": float(res.iloc[-1]["val AUROC"])}, indent=1, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
