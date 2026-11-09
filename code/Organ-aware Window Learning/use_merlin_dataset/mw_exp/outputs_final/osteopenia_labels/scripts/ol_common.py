"""Osteopenia label-handling comparison: shared data loading. Train/val only; test never read.
Labels: labels_trval.csv, split column filtered FIRST to train/val. geom15_levels.csv (all splits on disk) is
filtered to train/val ids immediately after reading; no test row is kept.
L1 trabecular HU = geom15 hu_body at L1 (1.5 mm seg15 vertebral body, mid-sagittal, eroded 2 px, median HU;
same feature as osteopenia_v15 'hu15_L1')."""
import os, numpy as np, pandas as pd
D = os.environ.get("OL_DATA", os.path.dirname(os.path.abspath(__file__)) + "/../data/")
if not D.endswith("/"): D += "/"


def load_base():
    lab = pd.read_csv(D + "labels_trval.csv", usecols=["study_id", "split", "y_op", "op_cat", "y_frac"])
    lab = lab[lab.split.isin(["train", "val"])]
    lab = lab.dropna(subset=["y_op"]).set_index("study_id")
    fold = pd.read_csv(D + "oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id").fold
    meta = pd.read_csv(D + "bundle_trval.csv", usecols=["study_id", "Age", "Gender"]).set_index("study_id")
    tr = list(lab.index[lab.split == "train"]); va = list(lab.index[lab.split == "val"])
    keep = set(tr) | set(va)
    g = pd.read_csv(D + "geom15_levels.csv", usecols=["id", "level", "hu_body"])
    g = g[g.id.isin(keep) & (g.level == "L1")].set_index("id").hu_body
    out = {}
    for name, ids in (("T", tr), ("V", va)):
        out[name] = pd.DataFrame(dict(y=lab.loc[ids, "y_op"].values.astype(int), cat=lab.loc[ids, "op_cat"].values,
                                      frac=lab.loc[ids, "y_frac"].values,
                                      age=meta.Age.reindex(ids).values.astype(float), sex=meta.Gender.reindex(ids).values,
                                      hu=g.reindex(ids).values.astype(float)), index=ids)
    out["T"]["fold"] = fold.loc[tr].values
    return out


def scheme_targets(T, scheme, k_soft=10.0, pu_g=None):
    """returns (target t in [0,1], weight w >= 0) for train rows."""
    y = T.y.values.astype(float); hu = T.hu.values; n = len(y)
    t, w = y.copy(), np.ones(n)
    if scheme == "A":
        pass
    elif scheme in ("B110", "B130"):
        thr = float(scheme[1:])
        w[(y == 0) & (hu < thr)] = 0.0                      # NaN HU -> comparison False -> kept
    elif scheme == "C":                                    # Elkan-Noto weighting, g = P(s=1|x) from scheme A OOF
        g, c = pu_g
        p = np.clip((1 - c) / c * g / np.clip(1 - g, 1e-6, None), 0, 1)
        t = np.where(y == 1, 1.0, p)
    elif scheme == "D":
        s = 1 / (1 + np.exp(-(110.0 - hu) / k_soft))
        s = np.where(np.isnan(hu), 0.0, s)
        t = np.maximum(y, s)
    else:
        raise ValueError(scheme)
    return t, w


def balanced_weights(t, w):
    """class-balanced weights for soft targets: returns per-row weight for the positive and negative component."""
    n = w.sum(); P = (t * w).sum(); N = ((1 - t) * w).sum()
    return w * t * n / (2 * P), w * (1 - t) * n / (2 * N)


def expand(X, t, w):
    """turn soft targets into duplicated weighted rows for sklearn (positive + negative copies)."""
    wp, wn = balanced_weights(t, w)
    ip, ineg = wp > 0, wn > 0
    Xe = np.concatenate([X[ip], X[ineg]]); ye = np.r_[np.ones(ip.sum()), np.zeros(ineg.sum())]
    we = np.r_[wp[ip], wn[ineg]]
    return Xe, ye, we
