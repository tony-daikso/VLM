"""BD4 pilot analysis -- evaluated ONLY inside the pilot set (never stacked on the full set: partial coverage would leak
the sampling design). Inverse-probability weights (stratum size / sampled) make AUROCs estimate the train population.
(a) single-feature weighted AUROC: native features vs their 3 mm counterparts
(b) repeated 5-fold CV inside the pilot: LR on [E3_late OOF + 3 mm porta/tube features] vs [... + native features]
    paired over the same splits; overall, mild positives vs all negatives, extrahepatic-only positives vs all negatives.
Usage: bd4_analyze.py WORKDIR OUTDIR"""
import sys, os, numpy as np, pandas as pd, warnings, json
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
W, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
p = pd.read_csv(f"{W}/bd4_pilot_ids.csv"); nat = pd.read_csv(f"{W}/bd4_native.csv")
lab = pd.read_csv(f"{W}/labels_trval.csv"); lab = lab[lab.split == "train"]
oof = pd.read_csv(f"{W}/oof_train.csv")[["study_id", "E3_late", "E0", "E3late_cat_+gb_duod"]]
f3 = pd.read_csv(f"{W}/bd3_all.csv"); hu = pd.read_csv(f"{W}/hu_all.csv")
d = p.merge(nat, on="study_id").merge(oof, on="study_id").merge(f3.drop(columns=[c for c in f3.columns if c in hu.columns and c != "study_id"]), on="study_id", how="left") \
     .merge(hu, on="study_id", how="left").merge(lab[["study_id", "kw_mild", "kw_marked", "bd_site", "y_chole", "kw_stent", "kw_pneumobilia"]], on="study_id")
ok = d.n_err.isna() if "n_err" in d else np.ones(len(d), bool)
print("pilot", len(p), "native ok", int(ok.sum()), "errors", d.n_err.value_counts().to_dict() if "n_err" in d else {})
d = d[ok].reset_index(drop=True)
y = d.y_bdd.values.astype(int); w = d.w.values
NAT = [c for c in nat.columns if c.startswith("n_") and c not in ("n_err", "n_dx", "n_dz", "n_shape", "n_err_tube")]
T3 = [c for c in f3.columns if c.startswith(("tube_", "hdl_", "duct_lp_", "phd_fl_"))]
H3 = ["porta_fluid_n", "porta_fluid_frac", "porta_fluid_maxcc", "porta_hu_med", "hil_low_n", "hil_low_maxcc", "phead_fluid_n",
      "liv_low_n", "liv_low_maxcc", "porta_air_n", "porta_metal_n"]
rows = []
def wauc(yy, s, ww):
    m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m], sample_weight=ww[m]) if 0 < yy[m].sum() < m.sum() else np.nan
for c in NAT + T3 + H3 + ["E3_late", "E3late_cat_+gb_duod"]:
    if c in d:
        rows.append(dict(kind="native" if c in NAT else "3mm", feature=c, wauc=wauc(y, d[c].values.astype(float), w)))
single = pd.DataFrame(rows); single["absdev"] = (single.wauc - .5).abs()
single.sort_values("absdev", ascending=False).to_csv(f"{O}/bd4_single_features.csv", index=False)
print(single.sort_values("absdev", ascending=False).head(25).to_string())
sev = np.where(d.kw_marked == 1, "marked", np.where(d.kw_mild == 1, "mild", "unstated"))
g_mild = (sev == "mild") & (y == 1); g_extra = (d.bd_site == "extrahepatic").values & (y == 1)


def prep(X):
    X = X.astype(float).copy()
    for c in X.columns:
        if c.endswith(("_ml", "_vol", "_n", "_maxcc", "_ncc", "_skel", "_ends", "_branch")):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.fillna(X.median())


C3 = ["porta_fluid_maxcc", "porta_fluid_frac", "hdl_fl_dmax", "hdl_fl_vol", "hdl_w_dmax", "duct_lp_dmax", "phd_fl_dmax",
      "tube_rel_t03_hil_ml", "tube_abs_t03_frac", "tube_rel_t03_ends", "hil_low_maxcc"]
CN = ["n_hdl_cbd_dmax", "n_hdl_dmax", "n_hdlw_cbd_dmax", "n_ph_cbd_dmax", "n_hdl_vol", "n_hdl_fluid_ml", "n_hil_dark_ml",
      "n_tube_hdl_t06_ml", "n_tube_hil_t06_ml", "n_tube_ph_t06_ml", "n_tube_hdl_t06_mscale"]
sets = {"base(E3_late)": ["E3_late"], "base+3mm_compact": ["E3_late"] + C3, "base+native_compact": ["E3_late"] + CN,
        "base+3mm_compact+native_compact": ["E3_late"] + C3 + CN, "base+3mm": ["E3_late"] + T3 + H3, "base+native": ["E3_late"] + NAT,
        "base+3mm+native": ["E3_late"] + T3 + H3 + NAT, "native only": NAT, "3mm only": T3 + H3}
REP = 20; res = {k: {"all": [], "mild": [], "extra": []} for k in sets}
for r in range(REP):
    skf = StratifiedKFold(5, shuffle=True, random_state=r)
    P = {k: np.zeros(len(d)) for k in sets}
    for a, b in skf.split(d, y):
        for k, cols in sets.items():
            X = prep(d[[c for c in cols if c in d]])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
            m.fit(X.iloc[a], y[a], logisticregression__sample_weight=w[a]); P[k][b] = m.decision_function(X.iloc[b])
    for k in sets:
        res[k]["all"].append(wauc(y, P[k], w))
        mm = g_mild | (y == 0); res[k]["mild"].append(wauc(y[mm], P[k][mm], w[mm]))
        mm = g_extra | (y == 0); res[k]["extra"].append(wauc(y[mm], P[k][mm], w[mm]))
out = []
for k in sets:
    rec = {"set": k}
    for g in ("all", "mild", "extra"):
        v = np.array(res[k][g]); rec[f"{g}_mean"] = v.mean(); rec[f"{g}_sd"] = v.std()
        if k != "base+3mm_compact":
            dv = v - np.array(res["base+3mm_compact"][g]); rec[f"{g}_d_vs_3mmc"] = dv.mean(); rec[f"{g}_d_frac_pos"] = (dv > 0).mean()
    out.append(rec)
R = pd.DataFrame(out); R.to_csv(f"{O}/bd4_pilot_cv.csv", index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string())
json.dump({"n_pilot_ok": int(len(d)), "n_pos": int(y.sum()), "n_mild_pos": int(g_mild.sum()), "n_extra_pos": int(g_extra.sum()),
           "sec_per_case_total_median": float(d.t_total.median()), "sec_load_median": float(d.t_load.median())},
          open(f"{O}/bd4_summary.json", "w"), indent=1)
print(open(f"{O}/bd4_summary.json").read())
