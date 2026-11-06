"""Experiment K evaluation: rib high-res path vs bone_ens / E3_late. Train+val only (table_parsed.pkl; test never read).

Inputs: --run DIR with preds_fold*.npz from rib_hr_train.py (train rows = outer-fold OOF, val rows = mean of fold models).
Fusion (fixed weight, chosen on TRAIN OOF only): fused = z(bone_ens) + w * z(rib_hr); z uses train-OOF mean/sd of the
same rows; w in a grid picked by train-OOF AUROC (rib target; any-fracture when the train subset has non-rib positives).
Report (val, only cases that have rib CPRs; references recomputed on the same set):
  rib AUROC (rib positives vs fracture negatives), rib-only AUROC, any-fracture AUROC, paired bootstrap deltas vs
  bone_ens and E3_late OOF, age-matched AUROC (+-2 y), Spearman(score, age | negatives); localization on train OOF
  positives (top rib side hit, level +-1 hit).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RIBS = [f"L{k}" for k in range(5, 13)] + [f"R{k}" for k in range(5, 13)]


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def matched_auc(y, p, age, tol=2.0):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    num = den = 0.0
    for i in pos:
        nb = neg[np.abs(age[neg] - age[i]) <= tol]
        if len(nb):
            num += (p[i] > p[nb]).sum() + 0.5 * (p[i] == p[nb]).sum()
            den += len(nb)
    return num / den if den else np.nan


def boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    d0 = auc(y, a) - auc(y, b)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    ds = []
    for _ in range(n):
        idx = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        ds.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return d0, lo, hi


def load_preds(run):
    fs = sorted(glob.glob(str(Path(run) / "preds_fold*.npz"))) + sorted(glob.glob(str(Path(run) / "preds_val_fold*.npz")))
    tr, va, R = {}, [], {}
    for f in fs:
        z = np.load(f)
        for i, s, l, r, p, k in zip(z["ids"], z["split"], z["logit"], z["rib_logit"], z["rib_present"], z["att_peak"]):
            if s == "train":
                tr[i] = l; R[i] = (r, p, k)
            else:
                va.append((i, l, r))
    v = pd.DataFrame(va, columns=["id", "logit", "r"])
    vm = v.groupby("id").logit.mean()
    vr = v.groupby("id").r.apply(lambda x: np.mean(np.stack(list(x)), 0))
    return pd.Series(tr), vm, R, vr, len(glob.glob(str(Path(run) / "preds_fold*.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out or a.run)
    t = pd.read_pickle(a.table)
    assert set(t.split) <= {"train", "val"}
    tr_s, va_s, Rtr, Rva, nf = load_preds(a.run)
    t["k"] = pd.concat([tr_s, va_s])
    t["e3"] = t.s_e3late
    t["be"] = t.s_bone_ens
    t["age"] = t.Age.astype(float)
    T = t[(t.split == "train") & t.k.notna()]
    V = t[(t.split == "val") & t.k.notna() & t.be.notna() & t.e3.notna()]
    res = {"n_folds": nf, "n_train_oof": int(len(T)), "n_val": int(len(V))}

    def z(col, ref):
        return (col - ref.mean()) / ref.std()
    # fusion weight on train OOF
    Tr = T[T.be.notna()]
    rib_tr = Tr[(Tr.rib == 1) | (Tr.y == 0)]
    grid = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    has_nonrib = int(((Tr.y == 1) & (Tr.rib != 1)).sum())
    sel = []
    for w in grid:
        f = z(Tr.be, Tr.be) + w * z(Tr.k, Tr.k)
        ar = auc((rib_tr.rib == 1).astype(int), f[rib_tr.index])
        aa = auc(Tr.y, f) if has_nonrib >= 100 else np.nan
        sel.append((w, ar, aa))
    crit = 2 if has_nonrib >= 100 else 1
    w_best = max(sel, key=lambda r: (r[crit] if r[crit] == r[crit] else -1))[0]
    res["fusion_grid_trainOOF"] = [dict(w=w, rib=r, anyfx=x) for w, r, x in sel]
    res["fusion_w"] = w_best
    res["fusion_criterion"] = "train OOF any-fracture" if crit == 2 else "train OOF rib (no non-rib positives in train subset)"
    for D in (T, V):
        D["fuse"] = z(D.be, Tr.be) + w_best * z(D.k, Tr.k)
    # fusion that adds the rib-hr score only for the rib channel inside a max: max(z(be), z(k)) (no extra weight)
    for D in (T, V):
        D["fmax"] = np.maximum(z(D.be, Tr.be), z(D.k, Tr.k))

    rows = []
    tgts = {"肋骨（vs 骨折陰性）": lambda D: D[(D.rib == 1) | (D.y == 0)].assign(yy=lambda d: (d.rib == 1).astype(int)),
            "只有肋骨（vs 骨折陰性）": lambda D: D[((D.rib == 1) & ~D.other_site_rib.astype(bool)) | (D.y == 0)].assign(
                yy=lambda d: (d.rib == 1).astype(int)),
            "任何骨折": lambda D: D.assign(yy=D.y.astype(int))}
    scores = {"rib_hr": "k", "bone_ens": "be", "E3_late": "e3", f"融合 w={w_best}": "fuse", "max(z)": "fmax"}
    for split, D in (("train_oof", T), ("val", V)):
        for tn, fn in tgts.items():
            d = fn(D)
            for sn, c in scores.items():
                if split == "train_oof" and sn in ("E3_late",):
                    pass
                dd = d[d[c].notna()]
                y, s, ag = dd.yy.values, dd[c].values, dd.age.values
                r = dict(split=split, target=tn, score=sn, n_pos=int(y.sum()), n=len(y), auroc=auc(y, s),
                         age_matched=matched_auc(y, s, ag), spearman_age_neg=spearmanr(s[y == 0], ag[y == 0], nan_policy="omit")[0])
                if split == "val" and sn not in ("bone_ens",):
                    for ref, rc in (("bone_ens", "be"), ("E3_late", "e3")):
                        if rc == c:
                            continue
                        d0, lo, hi = boot(y, s, dd[rc].values)
                        r[f"d_vs_{ref}"] = f"{d0:+.3f} [{lo:+.3f},{hi:+.3f}]"
                rows.append(r)
    df = pd.DataFrame(rows)
    # localization on train OOF positives
    loc = []
    for cid, row in T[T.rib == 1].iterrows():
        if cid not in Rtr:
            continue
        r, p, k = Rtr[cid]
        if not p.any():
            continue
        top = RIBS[int(np.argmax(np.where(p, r, -1e9)))]
        side = row.side if isinstance(row.side, str) else "?"
        lv = [(s, int(n)) for s, n in re.findall(r"([LRB])(\d+)", row.rib_levels or "")] if isinstance(row.rib_levels, str) else []
        e = dict(id=cid, top=top, side=side, levels=row.rib_levels)
        if side in ("L", "R"):
            e["side_hit"] = top[0] == side
        if lv:
            e["level_hit1"] = any(abs(int(top[1:]) - n) <= 1 for _, n in lv)
            e["side_level_hit1"] = any(abs(int(top[1:]) - n) <= 1 and (s == "B" or s == top[0]) for s, n in lv)
        loc.append(e)
    L = pd.DataFrame(loc)
    res["loc"] = {"n_side": int(L.side_hit.notna().sum()) if "side_hit" in L else 0,
                  "side_hit": float(L.side_hit.dropna().mean()) if "side_hit" in L else None,
                  "n_level": int(L.level_hit1.notna().sum()) if "level_hit1" in L else 0,
                  "level_hit1": float(L.level_hit1.dropna().mean()) if "level_hit1" in L else None,
                  "side_level_hit1": float(L.side_level_hit1.dropna().mean()) if "side_level_hit1" in L else None}
    # random baseline for level+-1 given 8 levels 5..12 (uniform top over present ribs ~ 3/8 interior)
    tag = a.tag
    df.to_csv(out / f"eval{tag}.csv", index=False)
    L.to_csv(out / f"loc{tag}.csv", index=False)
    json.dump(res, open(out / f"eval{tag}.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps(res, indent=1, default=float))
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
