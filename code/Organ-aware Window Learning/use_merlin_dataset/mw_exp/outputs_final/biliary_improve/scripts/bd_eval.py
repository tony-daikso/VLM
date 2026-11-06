"""Shared evaluation for the biliary-dilation improvement steps (BD1-BD5). TRAIN 5-fold OOF for selection, VAL only
for reporting. Metrics: AUROC + bootstrap 95% CI, paired-bootstrap delta vs baseline, age-matched AUROC (5-year bins,
weighted by pos*neg pairs), subgroups, sensitivity at 90% specificity. Labels unchanged (yes/probably_yes = 1)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

B = 1000


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[m], s[m]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def boot_ci(y, s, seed=0, b=B):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(b):
        i = rng.integers(0, n, n); v.append(auc(y[i], s[i]))
    return np.nanpercentile(v, [2.5, 97.5])


def paired_delta(y, s, s0, seed=0, b=B):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(b):
        i = rng.integers(0, n, n); v.append(auc(y[i], s[i]) - auc(y[i], s0[i]))
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return auc(y, s) - auc(y, s0), lo, hi, float(np.mean(np.array(v) <= 0))


def age_matched(y, s, age):
    num = den = 0.0
    ab = np.floor(np.asarray(age, float) / 5)
    for a in np.unique(ab[~np.isnan(ab)]):
        k = ab == a; yy = y[k]
        if 0 < yy.sum() < len(yy):
            w = yy.sum() * (len(yy) - yy.sum()); num += auc(yy, s[k]) * w; den += w
    return num / den if den else np.nan


def sens_at_spec(y, s, spec=0.90):
    thr = np.quantile(s[y == 0], spec)
    return float((s[y == 1] > thr).mean())


def add_groups(df):
    df = df.copy()
    df["sev"] = np.where(df.kw_marked == 1, "moderate/marked", np.where(df.kw_mild == 1, "mild", "unstated"))
    df["g_mild"] = df.sev == "mild"
    df["g_extra"] = df.bd_site == "extrahepatic"
    df["g_chole"] = df.y_chole == 1
    df["g_stent"] = (df.kw_stent == 1) | (df.kw_pneumobilia == 1)
    return df


def full_metrics(df, col, base=None, seed=0):
    """df: rows of one split with y_bdd, Age, groups, score columns. Returns dict of metrics."""
    y = df.y_bdd.values.astype(int); s = df[col].values.astype(float)
    r = {"auc": auc(y, s)}
    r["ci_lo"], r["ci_hi"] = boot_ci(y, s, seed)
    r["age_matched"] = age_matched(y, s, df.Age.values)
    r["sens90"] = sens_at_spec(y, s)
    neg = y == 0
    for g, nm in (("g_mild", "mild"), ("g_extra", "extra_only")):
        k = (df[g].values & (y == 1)) | neg
        r[f"auc_{nm}"] = auc(y[k], s[k])
    for g, nm in (("g_chole", "chole"), ("g_stent", "stent")):
        for v, t in ((True, "yes"), (False, "no")):
            k = df[g].values == v
            r[f"auc_{nm}_{t}"] = auc(y[k], s[k])
    k = (~df.g_stent.values) | (y == 1)                    # all positives vs negatives WITHOUT stent/air
    r["auc_stentneg_vs_allpos"] = auc(y[(df.g_stent.values & neg) | (y == 1)], s[(df.g_stent.values & neg) | (y == 1)])
    if base is not None and base != col:
        d, lo, hi, p = paired_delta(y, s, df[base].values.astype(float), seed)
        r.update(delta=d, d_lo=lo, d_hi=hi, p_le0=p)
    return r


def fmt(r, delta=True):
    s = f"{r['auc']:.4f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
    if delta and "delta" in r:
        s += f"  Δ {r['delta']:+.4f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]"
    return s
