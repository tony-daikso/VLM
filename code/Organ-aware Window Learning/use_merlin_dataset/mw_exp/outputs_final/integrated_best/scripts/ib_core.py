"""Integrated-best evaluation core (train/val only; test is never read).

Shared by every finding so that all rows of the integrated table use the same rules:
  * candidates: LR (C 0.1/0.3/1, class-balanced, standardised, log1p on heavy-tailed non-negative columns)
    and 4 HistGradientBoosting settings (same grid as gallstones_improve/gs_stack2.py);
  * model choice = max train 5-fold OOF AUROC, using the finding's own folds (the folds of its E3_late OOF);
  * the chosen model is refit on all train rows and applied once to val;
  * val: AUROC + stratified bootstrap CI (2,000), paired Δ vs the expB E3_late val score (official baseline,
    3-seed mean prob) and vs the re-run E3_late of the same analysis; age-matched AUROC (10-y bins, pair weighted);
    Spearman(score, age) on negatives; sensitivity at 90 % specificity (val-own threshold and train-OOF threshold).
  * age / sex are never model inputs.
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

EDGES = (0, 30, 40, 50, 60, 70, 80, 200)


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); m = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[m], s[m]; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def age_matched_auc(y, s, age):
    y = np.asarray(y, float); s = np.asarray(s, float); age = np.asarray(age, float)
    b = pd.cut(pd.Series(age), EDGES, right=False).values; num = den = 0.0
    for g in pd.unique(b[~pd.isna(b)]):
        m = np.asarray(b == g); n1 = y[m].sum(); n0 = m.sum() - n1
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den if den else np.nan


def spearman_neg(y, s, age):
    m = (np.asarray(y) == 0) & ~np.isnan(np.asarray(age, float)) & ~np.isnan(np.asarray(s, float))
    return float(spearmanr(np.asarray(s)[m], np.asarray(age)[m]).correlation)


def sens_at_spec(y, s, thr=None, spec=0.9):
    y = np.asarray(y); s = np.asarray(s, float)
    if thr is None: thr = np.quantile(s[y == 0], spec)
    return float((s[y == 1] > thr).mean()), float(thr)


def _bidx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); a = [auc(y[ix], s[ix]) for ix in (_bidx(y, rng) for _ in range(n))]
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def boot_delta(y, a, b, n=2000, seed=0, age=None):
    """paired stratified bootstrap of AUROC(b) - AUROC(a) (age-matched AUROC if age is given)."""
    rng = np.random.default_rng(seed); d = []
    f = (lambda ix, s: auc(y[ix], s[ix])) if age is None else (lambda ix, s: age_matched_auc(y[ix], s[ix], age[ix]))
    for _ in range(n):
        ix = _bidx(y, rng); d.append(f(ix, b) - f(ix, a))
    d = np.array(d)
    return dict(mean=float(np.nanmean(d)), ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                p_le0=float(np.nanmean(d <= 0)))


# ------------------------------------------------------------------ candidates
def LR(C): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
def GB(lr, leaves, msl, it): return lambda: HistGradientBoostingClassifier(
    max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=1.0, random_state=0)
CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200), "GBM_lr.02_l15_m60_800": GB(0.02, 15, 60, 800)}


def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


class Prep:
    """column-wise transform fitted on train: log1p for heavy-tailed non-negative columns, median imputation."""
    def fit(self, X):
        X = X.astype(float); self.cols = list(X.columns); self.log = []
        for c in self.cols:
            v = X[c].dropna()
            if len(v) and v.min() >= 0 and (v.max() > 50) and (v.skew() > 2): self.log.append(c)
        self.med = self._log(X).median(); return self
    def _log(self, X):
        X = X[self.cols].astype(float).copy()
        for c in self.log: X[c] = np.log1p(X[c].clip(lower=0))
        return X
    def transform(self, X): return self._log(X).fillna(self.med).fillna(0.0)


def select_and_fit(T, V, y, folds, featsets, cands=None, log=print):
    """T/V: frames with all candidate columns; featsets: {name: [cols]}. Returns dict with chosen model, OOF, val score."""
    cands = cands or CANDS; res = {}; best = None
    for fs, cols in featsets.items():
        pr = Prep().fit(T[cols]); XT = pr.transform(T[cols]).values
        for cn, mk in cands.items():
            oof = np.zeros(len(T))
            for k in np.unique(folds):
                tr, te = folds != k, folds == k
                m = mk().fit(XT[tr], y[tr]); oof[te] = score(m, XT[te])
            a = auc(y, oof); res[f"{fs}|{cn}"] = a
            log(f"    {fs:<40s} {cn:<24s} OOF {a:.4f}")
            if best is None or a > best[0]: best = (a, fs, cn, oof)
    a, fs, cn, oof = best; cols = featsets[fs]
    pr = Prep().fit(T[cols]); m = CANDS[cn]().fit(pr.transform(T[cols]).values, y)
    sv = score(m, pr.transform(V[cols]).values)
    _, thr = sens_at_spec(y, oof)
    return dict(featset=fs, cand=cn, cols=cols, oof=oof, val=sv, model=m, prep=pr, cv=res, oof_auc=a, thr90_oof=thr)


def evaluate(name, yT, oofT, baseT, ageT, yV, sV, baseV_expB, baseV_rerun, ageV, thr_oof=None, nboot=2000):
    """uniform metrics for one finding (baseline = expB E3_late on val; re-run E3_late on train OOF)."""
    yT = np.asarray(yT, int); yV = np.asarray(yV, int)
    oofT, baseT, sV = (np.asarray(x, float) for x in (oofT, baseT, sV))
    baseV_expB = np.asarray(baseV_expB, float); ageT = np.asarray(ageT, float); ageV = np.asarray(ageV, float)
    r = dict(finding=name, n_tr=len(yT), pos_tr=int(yT.sum()), n_val=len(yV), pos_val=int(yV.sum()))
    r["oof"] = auc(yT, oofT); r["oof_E3"] = auc(yT, baseT)
    r["d_oof"] = boot_delta(yT, baseT, oofT, n=nboot)
    r["oof_age"] = age_matched_auc(yT, oofT, ageT); r["oof_age_E3"] = age_matched_auc(yT, baseT, ageT)
    r["val"] = auc(yV, sV); r["val_ci"] = boot_ci(yV, sV, n=nboot)
    r["val_E3_expB"] = auc(yV, baseV_expB); r["val_E3_expB_ci"] = boot_ci(yV, baseV_expB, n=nboot)
    r["d_val_expB"] = boot_delta(yV, baseV_expB, sV, n=nboot)
    if baseV_rerun is not None:
        bvr = np.asarray(baseV_rerun, float); r["val_E3_rerun"] = auc(yV, bvr); r["d_val_rerun"] = boot_delta(yV, bvr, sV, n=nboot)
    r["val_age"] = age_matched_auc(yV, sV, ageV); r["val_age_E3"] = age_matched_auc(yV, baseV_expB, ageV)
    r["d_val_age"] = boot_delta(yV, baseV_expB, sV, n=500, age=ageV)
    r["spearman_neg"] = spearman_neg(yV, sV, ageV); r["spearman_neg_E3"] = spearman_neg(yV, baseV_expB, ageV)
    r["sens90_val"], _ = sens_at_spec(yV, sV); r["sens90_val_E3"], _ = sens_at_spec(yV, baseV_expB)
    if thr_oof is not None: r["sens90_val_oofthr"], _ = sens_at_spec(yV, sV, thr_oof)
    r["sens90_oof"], _ = sens_at_spec(yT, oofT)
    r["flag_age_drop"] = bool(r["val_age"] < r["val_age_E3"])
    return r
