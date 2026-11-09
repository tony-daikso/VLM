"""GS2 pilot analysis (TRAIN only): per-feature AUROC of native-resolution GB features, and their added value on top of the
GS1 stack (train OOF score) with pilot-level repeated 5-fold CV. Inverse-probability weights (stratified sample) -> population-weighted AUROC.
usage: gs2_analyze.py pilot_ids.csv native.csv gs1_oof_train.csv gs1_feat3mm.csv out_dir"""
import sys, os, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
P, N, S, G, O = sys.argv[1:6]; os.makedirs(O, exist_ok=True)
p = pd.read_csv(P); nat = pd.read_csv(N); s = pd.read_csv(S); g = pd.read_csv(G)
best = [c for c in s.columns if c.startswith("V4")][0]
d = p.merge(nat, on="study_id", how="left").merge(s[["study_id", "E3_late", "quickwins", best]], on="study_id", how="left").merge(
    g[["study_id", "g1_max_minus_med", "g1_f_hi40"]], on="study_id", how="left")
d = d.rename(columns={best: "gs1"}).reset_index(drop=True)
y = d.y_gs.values.astype(int); w = d.w.values
NF = [c for c in nat.columns if c.startswith("n_") and c not in ("n_err", "n_dx", "n_dz", "n_gb3_vox") and pd.api.types.is_numeric_dtype(nat[c])]
print("pilot", len(d), "pos", y.sum(), "native ok", int(d.n_m_max.notna().sum()), "errors", d.get("n_err", pd.Series(dtype=str)).value_counts().to_dict(), flush=True)

def wauc(yy, ss, ww):
    k = ~np.isnan(ss)
    return float(roc_auc_score(yy[k], ss[k], sample_weight=ww[k]))
gbp = d.n_m_max.notna().values  # GB present with native features
nonvis_pos = (d.stratum == "pos_nonvisible3mm").values
neg_gb = (y == 0) & gbp
rows = []
for c in NF + ["gs1", "E3_late", "quickwins"]:
    x = d[c].astype(float).values
    if np.isnan(x).all() or np.nanstd(x) == 0: continue
    r = {"feature": c}
    for nm, msk in (("all_GBpresent", gbp), ("nonvis_pos_vs_neg", nonvis_pos | neg_gb), ("vis_pos_vs_neg", (d.stratum == "pos_visible3mm").values | neg_gb)):
        k = msk & ~np.isnan(x)
        a = wauc(y[k], x[k], w[k]) if len(np.unique(y[k])) == 2 else np.nan
        r[nm] = a
    rows.append(r)
F = pd.DataFrame(rows); F["dir"] = np.where(F.all_GBpresent >= .5, "+", "-")
for c in ("all_GBpresent", "nonvis_pos_vs_neg", "vis_pos_vs_neg"):
    F[c + "_abs"] = np.where(F.all_GBpresent >= .5, F[c], 1 - F[c])
F = F.sort_values("nonvis_pos_vs_neg_abs", ascending=False); F.to_csv(f"{O}/gs2_pilot_feature_auroc.csv", index=False)
print(F.head(25).round(3).to_string(), flush=True)

# descriptive: native hyperdense focus rates per stratum
desc = d.groupby("stratum").agg(n=("y_gs", "size"), native_ok=("n_m_max", lambda x: int(x.notna().sum())),
                                max_med=("n_m_max", "median"), f_ml130=("n_md_ml130", lambda x: float((x > 0.01).mean())),
                                f_cc200=("n_md_cc200", lambda x: float((x >= 5).mean())), f_max400=("n_md_max", lambda x: float((x > 400).mean())),
                                hypo_cc_med=("n_hypo_cc", "median"), dog3_min_med=("n_dog3_min", "median"), dep_med=("n_dep_diff", "median"))
desc.to_csv(f"{O}/gs2_pilot_strata.csv"); print(desc.round(3).to_string(), flush=True)

# added value: repeated stratified 5-fold CV within the pilot
def logf(X):
    X = X.copy()
    for c in X.columns:
        if "_ml" in c or "_cc" in c or c.endswith("_vox"): X[c] = np.log1p(X[c].clip(lower=0))
    return X
SETS = {"gs1_only": ["gs1"], "gs1+native(all)": ["gs1"] + NF,
        "gs1+native(精簡4)": ["gs1", "n_md_max", "n_m_cc200", "n_md_cc400", "n_dog4_min"],
        "gs1+native(核心)": ["gs1", "n_md_max", "n_md_p999", "n_md_ml130", "n_md_ml200", "n_md_cc200", "n_hypo_cc", "n_hypo40_cc", "n_dog3_min", "n_dog3_max", "n_dog2_ml_neg15", "n_dep_diff", "n_in_sm_std", "n_bile_med"]}
MODELS = {"LR_C0.1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
          "LR_C0.01": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")),
          "GBM_small": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0, random_state=0)}
R = 10; preds = {}
for sn, cols in SETS.items():
    for mn, mk in MODELS.items():
        if sn == "gs1_only" and mn != "LR_C0.1": continue
        P_ = np.zeros((R, len(d)))
        for r in range(R):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(d, y):
                Xtr = logf(d.loc[tr, cols].astype(float)); med = Xtr.median(); Xte = logf(d.loc[te, cols].astype(float)).fillna(med); Xtr = Xtr.fillna(med)
                m = mk().fit(Xtr, y[tr]); P_[r, te] = m.decision_function(Xte) if hasattr(m, "decision_function") else m.predict_proba(Xte)[:, 1]
        preds[f"{sn}|{mn}"] = P_
base = preds["gs1_only|LR_C0.1"]
res = {}
rng = np.random.default_rng(0)
for k, P_ in preds.items():
    a_all = np.mean([wauc(y, P_[r], w) for r in range(R)])
    m_nv = nonvis_pos | (y == 0)
    a_nv = np.mean([wauc(y[m_nv], P_[r][m_nv], w[m_nv]) for r in range(R)])
    a_nv_b = np.mean([wauc(y[m_nv], base[r][m_nv], w[m_nv]) for r in range(R)])
    # bootstrap CI of the delta vs gs1_only (mean over repeats), stratified resampling, weighted AUROC
    pm, bm = P_.mean(0), base.mean(0); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; dl = []
    for _ in range(1000):
        ix = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); dl.append(wauc(y[ix], pm[ix], w[ix]) - wauc(y[ix], bm[ix], w[ix]))
    mn_ = nonvis_pos | (y == 0); i1n, i0n = np.where((y == 1) & mn_)[0], np.where((y == 0) & mn_)[0]; dn = []
    for _ in range(1000):
        ix = np.concatenate([rng.choice(i1n, len(i1n)), rng.choice(i0n, len(i0n))]); dn.append(wauc(y[ix], pm[ix], w[ix]) - wauc(y[ix], bm[ix], w[ix]))
    res[k] = dict(delta_nonvis_ci=[float(np.percentile(dn, 2.5)), float(np.percentile(dn, 97.5))], w_auc_all=a_all, w_auc_nonvis=a_nv, delta_all=a_all - np.mean([wauc(y, base[r], w) for r in range(R)]),
                  delta_nonvis=a_nv - a_nv_b, delta_all_ci=[float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                  unweighted_all=float(np.mean([roc_auc_score(y, P_[r]) for r in range(R)])))
    print(k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b]) for a, b in res[k].items()}, flush=True)
json.dump(res, open(f"{O}/gs2_pilot_cv.json", "w"), ensure_ascii=False, indent=1, default=float)
print("timing (s/case, 3 procs on Mac, Passport HDD):", d.t_total.describe().round(2).to_dict())
