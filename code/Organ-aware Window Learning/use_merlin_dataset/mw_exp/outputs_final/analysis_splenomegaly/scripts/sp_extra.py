"""Follow-up on TRAIN OOF (val descriptive): body-size ablation without age/sex, label-boundary groups, val bootstrap.
Usage: sp_extra.py <workdir>  (imports the frames built by sp_analyze.py by re-running its data prep)"""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
D = sys.argv[1]; O = D + "/out/"
src = open(os.path.join(os.path.dirname(__file__), "sp_analyze.py")).read().split("# ---------- 1. geometry")[0]
exec(src)
fold = T.fold.values
def gbm(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
def cv(cols, base=None):
    X = T[cols].values.astype(float)
    if base: X = np.c_[X, T[base].values]
    out = np.zeros(len(T))
    for k in range(5):
        a, b = fold != k, fold == k
        out[b] = gbm().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return out
def val(cols, base=None):
    X = T[cols].values.astype(float); Xv = V[cols].values.astype(float)
    if base: X = np.c_[X, T[base].values]; Xv = np.c_[Xv, V[base].values]
    return gbm().fit(X, y).predict_proba(Xv)[:, 1]
GEO = ["log_ml", "sp_cc_cm", "sp_pc1_cm", "sp_pc2_cm", "sp_pc3_cm", "sp_max_axial_cm2", "sp_ap_cm", "sp_tr_cm", "cranial_cut", "caudal_cut", "absent"]
SIZE = ["vert_cm2", "body_cm2", "L1_h_cm", "L1_body_ap_cm", "L1_body_w_cm", "fov_z_cm"]
sets = {"geom": GEO, "geom+體型(不含年齡性別)": GEO + SIZE, "geom+體型+肝腎體積": GEO + SIZE + ["liver_ml", "kid_ml"],
        "geom+肝體積": GEO + ["liver_ml"], "geom+腎體積": GEO + ["kid_ml"], "geom+年齡性別": GEO + ["Age", "male"],
        "geom+體型+肝腎+年齡性別": GEO + SIZE + ["liver_ml", "kid_ml", "Age", "male"]}
rows = []; rng = np.random.default_rng(0)
bs = [rng.integers(0, len(V), len(V)) for _ in range(1000)]
def A(yy, s): return roc_auc_score(yy, s)
for nm, cols in sets.items():
    for base in [None, "E3_late"]:
        o = cv(cols, base); v = val(cols, base)
        r = {"features": nm, "with_E3_late": base is not None, "train_oof": A(y, o), "val_desc": A(yv, v)}
        if base:
            d = [A(yv[i], v[i]) - A(yv[i], V.E3_late.values[i]) for i in bs]
            r["val_delta_vs_E3_late"] = r["val_desc"] - A(yv, V.E3_late); r["val_delta_ci_lo"], r["val_delta_ci_hi"] = np.percentile(d, [2.5, 97.5])
            # age-matched style check: AUROC within age strata (mean over 4 strata)
            T["_s"] = o
        rows.append(r); print(r, flush=True)
abl = pd.DataFrame(rows); abl.to_csv(O + "stack_ablation_bodysize.csv", index=False)
# correlations of size proxies with spleen volume in train negatives
cr = T.loc[y == 0, ["log_ml"] + SIZE + ["liver_ml", "kid_ml", "Age", "male"]].corr()["log_ml"].round(3).to_dict()
# label boundary: probably_no / borderline wording
lb = {}
for nm, m in {"probably_no": T.sp_cat == "probably_no", "no": T.sp_cat == "no", "not_mentioned": T.sp_cat == "not_mentioned",
              "yes": T.sp_cat == "yes", "yes+borderline": (T.sp_cat == "yes") & (T.q_kw_borderline == 1), "yes+mild": (T.sp_cat == "yes") & (T.q_kw_mild == 1)}.items():
    lb[nm] = {"n": int(m.sum()), "vol_median": float(T.sp_ml[m].median()), "vol_p25": float(T.sp_ml[m].quantile(.25)), "vol_p75": float(T.sp_ml[m].quantile(.75)),
              "frac_borderline_kw": float(T.kw_borderline[m].mean()), "frac_mild_kw": float(T.kw_mild[m].mean()),
              "E3_rank_pct_median": float(T.E3_late.rank(pct=True)[m].median())}
base = A(y, T.E3_late)
keep = (T.sp_cat != "probably_no").values
lb["drop_probably_no_gain"] = A(y[keep], T.E3_late[keep]) - base
# grey zone: volume between neg p90 and pos p25
g = (T.sp_ml >= T.sp_ml[y == 0].quantile(.9)) & (T.sp_ml <= T.sp_ml[y == 1].quantile(.25))
lb["grey_zone"] = {"n": int(g.sum()), "pos": int(y[g].sum()), "prev": float(y[g].mean()), "E3_auroc_inside": A(y[g], T.E3_late[g]),
                   "vol_auroc_inside": A(y[g], T.log_ml[g]), "frac_of_all_pos": float(y[g].sum() / y.sum()), "frac_of_all_neg": float((g & (y == 0)).sum() / (y == 0).sum()),
                   "auroc_without_grey": A(y[~g], T.E3_late[~g])}
# reported length vs label (only where the report gives a length)
m = T.len_cm_report.notna()
lb["report_length"] = {"n": int(m.sum()), "pos": int(y[m].sum()), "auroc_reported_len": A(y[m], T.len_cm_report[m]) if 0 < y[m].sum() < m.sum() else None,
                       "pos_reported_lt13": int(((T.len_cm_label < 13) & (y == 1)).sum()), "neg_reported_ge13": int(((T.len_cm_report >= 13) & (y == 0)).sum())}
# within-sex volume quantiles of negatives
lb["neg_vol_p95_by_sex"] = T[y == 0].groupby("Gender").sp_ml.quantile(.95).round(0).to_dict()
lb["pos_rate_by_sex"] = T.groupby("Gender").y_sp.mean().round(4).to_dict()
# train vs val difficulty: volume-only AUROC per split + mix
lb["split_difficulty"] = {"train_vol_only": A(y, T.log_ml), "val_vol_only": A(yv, V.log_ml),
                          "train_frac_notmentioned_neg": float((T.sp_cat == "not_mentioned").mean()), "val_frac_notmentioned_neg": float((V.sp_cat == "not_mentioned").mean()),
                          "train_frac_mild_pos": float(T.q_kw_mild[y == 1].mean()), "val_frac_mild_pos": float(V.q_kw_mild[yv == 1].mean()),
                          "train_pos_median_ml": float(T.sp_ml[y == 1].median()), "val_pos_median_ml": float(V.sp_ml[yv == 1].median()),
                          "train_frac_probably_no": float((T.sp_cat == "probably_no").mean()), "val_frac_probably_no": float((V.sp_cat == "probably_no").mean())}
json.dump({"size_corr_with_logml_in_neg": cr, "label_boundary": lb}, open(O + "extra.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"size_corr": cr, "lb": lb}, ensure_ascii=False, indent=1, default=float))
