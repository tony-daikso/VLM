"""AAA error analysis on TRAIN 5-fold OOF (val descriptive only; test never loaded).
Inputs (same dir as this script's parent): labels_trval.csv, meta_trval.csv, oof_train.csv, scores_val_descriptive.csv,
expB_val_aaa_probs.csv, geom_a6.csv, geom_ct.csv.  Outputs: results_train_oof.csv, per_case_train_oof.csv, stdout log."""
import os, sys, warnings
warnings.filterwarnings("ignore")
try: os.nice(19)
except OSError: pass
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
rng = np.random.default_rng(0)
NB = 2000
lab = pd.read_csv(D + "labels_trval.csv")
meta = pd.read_csv(D + "meta_trval.csv")
geo = pd.concat([pd.read_csv(D + "geom_a6.csv"), pd.read_csv(D + "geom_ct.csv")]).drop_duplicates("study_id")
df = lab.merge(meta, on="study_id", how="left").merge(geo, on="study_id", how="left")
df["male"] = (df.Gender == "Male").astype(float)
df["contrast_b"] = df.contrast.astype(str).str.lower().eq("true").astype(float)
T = df[df.split == "train"].merge(pd.read_csv(D + "oof_train.csv"), on="study_id")
T = T[T.y_aaa.notna()].reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(D + "scores_val_descriptive.csv"), on="study_id")
V = V.merge(pd.read_csv(D + "expB_val_aaa_probs.csv").rename(columns={"E0": "expB_E0", "E3_late": "expB_E3"}), on="study_id", how="left")
V = V[V.y_aaa.notna()].reset_index(drop=True)
y = T.y_aaa.values.astype(int); yv = V.y_aaa.values.astype(int)
print(f"train N={len(T)} pos={y.sum()} | val N={len(V)} pos={yv.sum()} | geom coverage train {T.infra_eq_rmax.notna().mean():.3f}")
R = []
def auc(yy, s):
    m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
def boot(yy, s, s2=None, nb=NB):
    """stratified bootstrap (pos and neg resampled separately); returns (lo, hi) of AUROC or of AUROC(s)-AUROC(s2)."""
    m = ~np.isnan(s) if s2 is None else ~(np.isnan(s) | np.isnan(s2))
    yy, s = yy[m], s[m]; s2 = s2[m] if s2 is not None else None
    P, N = np.where(yy == 1)[0], np.where(yy == 0)[0]
    out = []
    for _ in range(nb):
        i = np.r_[rng.choice(P, len(P)), rng.choice(N, len(N))]
        a = roc_auc_score(yy[i], s[i])
        if s2 is not None: a -= roc_auc_score(yy[i], s2[i])
        out.append(a)
    return np.percentile(out, [2.5, 97.5])
def rec(sec, item, npos, nneg, val, lo=np.nan, hi=np.nan, note=""):
    R.append(dict(section=sec, item=item, n_pos=int(npos), n_neg=int(nneg), value=round(float(val), 4) if pd.notna(val) else np.nan,
                  ci_lo=round(float(lo), 4) if pd.notna(lo) else np.nan, ci_hi=round(float(hi), 4) if pd.notna(hi) else np.nan, note=note))
    print(f"[{sec}] {item}: {val:.4f} [{lo:.4f}, {hi:.4f}] (pos {npos}, neg {nneg}) {note}")
def full(sec, item, yy, s, note=""):
    a = auc(yy, s); lo, hi = boot(yy, s)
    rec(sec, item, (yy[~np.isnan(s)] == 1).sum(), (yy[~np.isnan(s)] == 0).sum(), a, lo, hi, note)
    return a
def delta(sec, item, yy, s, s2, note=""):
    m = ~(np.isnan(s) | np.isnan(s2)); a = auc(yy[m], s[m]) - auc(yy[m], s2[m]); lo, hi = boot(yy, s, s2)
    rec(sec, item, yy[m].sum(), (yy[m] == 0).sum(), a, lo, hi, note)

# ------------------------------------------------------------------ 1. reliability
S0, S3 = T.E0.values, T.E3_late.values
full("1_overall", "train OOF E0", y, S0)
full("1_overall", "train OOF E3_late", y, S3)
for c in [c for c in T.columns if c.startswith("E1_") and not c.endswith("_ath")]:
    full("1_overall", "train OOF " + c, y, T[c].values)
delta("1_overall", "train OOF E3_late - E0", y, S3, S0)
full("1_overall", "val expB E0 (descriptive)", yv, V.expB_E0.values)
full("1_overall", "val expB E3_late (descriptive)", yv, V.expB_E3.values)
full("1_overall", "val this-recipe E3_late (descriptive)", yv, V.E3_late.values)
# leverage of a single positive at N_pos=32 / 160
neg = np.sort(V.expB_E3.values[yv == 0]); rv = np.searchsorted(neg, V.expB_E3.values[yv == 1]) / len(neg)
print("val positives' rank among negatives (sorted):", np.round(np.sort(rv), 3).tolist())
rec("1_reliability", "val: AUROC loss contributed by worst 3 positives", 3, len(neg), np.sort(1 - rv)[-3:].sum() / len(rv))
rec("1_reliability", "val: AUROC if worst positive removed", yv.sum() - 1, len(neg), (np.sort(rv)[1:]).mean())
negt = np.sort(S3[y == 0]); rt = np.searchsorted(negt, S3[y == 1]) / len(negt)
T["rank_E3"] = np.where(y == 1, np.searchsorted(negt, S3) / len(negt), np.nan)
# subsample simulation: how variable is AUROC at 32 positives drawn from train OOF?
sim = [auc(np.r_[np.ones(32), np.zeros(5000)], np.r_[rng.choice(S3[y == 1], 32, replace=False), rng.choice(S3[y == 0], 5000, replace=False)]) for _ in range(2000)]
rec("1_reliability", "train-OOF E3 subsampled to 32 pos / 5000 neg: SD of AUROC", 32, 5000, np.std(sim), *np.percentile(sim, [2.5, 97.5]))

# ------------------------------------------------------------------ 2. geometry
GEO = ["infra_eq_rmax", "infra_minor_rmax", "infra_feret_rmax", "infra_major_rmax", "abd_eq_rmax", "abd_minor_rmax",
       "all_eq_rmax", "thor_eq_rmax", "infra_ratio", "infra_vol_ml", "infra_grow_eq_rmax", "il_eq_rmax", "infra_eq_max"]
for g in GEO:
    full("2_geometry_alone", g, y, T[g].values.astype(float))
for thr in (25, 27.5, 30, 32.5, 35):
    s = T.infra_eq_rmax.values; m = ~np.isnan(s)
    se = (s[m & (y == 1)] >= thr).mean(); sp = (s[m & (y == 0)] < thr).mean()
    rec("2_threshold", f"infra eq diameter >= {thr} mm: sensitivity", (m & (y == 1)).sum(), (m & (y == 0)).sum(), se, note=f"specificity {sp:.4f}")
# measured vs reported diameter
q = T.q_diam_cm.values * 10; mm = (y == 1) & ~np.isnan(q) & ~np.isnan(T.infra_eq_rmax.values)
if mm.sum() > 5:
    for g in ("infra_eq_rmax", "infra_minor_rmax", "infra_feret_rmax", "abd_eq_rmax", "infra_grow_eq_rmax"):
        d = T[g].values[mm] - q[mm]
        rec("2_meas_vs_report", f"{g} - reported diameter (mm), positives with stated size", mm.sum(), 0, np.median(d),
            *np.percentile(d, [25, 75]), note=f"spearman {spearmanr(T[g].values[mm], q[mm])[0]:.3f} (CI cols = IQR)")
# stacking (same folds as OOF)
fold = T.fold.values
def cv(X, kind="lr"):
    p = np.full(len(T), np.nan)
    for k in range(5):
        a, b = fold != k, fold == k
        if kind == "lr":
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
            Xa = np.nan_to_num(X[a], nan=np.nanmedian(X[a], 0)); Xb = np.nan_to_num(X[b], nan=np.nanmedian(X[a], 0))
            mdl.fit(Xa, y[a]); p[b] = mdl.predict_proba(Xb)[:, 1]
        else:
            mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40,
                                                 l2_regularization=1.0, class_weight="balanced", random_state=0)
            mdl.fit(X[a], y[a]); p[b] = mdl.predict_proba(X[b])[:, 1]
    return p
def fitval(X, Xv, kind="lr"):
    if kind == "lr":
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
        med = np.nanmedian(X, 0); mdl.fit(np.nan_to_num(X, nan=med), y); return mdl.predict_proba(np.nan_to_num(Xv, nan=med))[:, 1]
    mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    mdl.fit(X, y); return mdl.predict_proba(Xv)[:, 1]
for dd in (T, V):
    dd["log_infra"] = np.log(dd.infra_eq_rmax); dd["log_abd"] = np.log(dd.abd_eq_rmax); dd["log_minor"] = np.log(dd.infra_minor_rmax)
    dd["metal"] = np.log1p(dd.v_n1600.astype(float)); dd["log_il"] = np.log(dd.il_eq_rmax)
G1 = ["log_infra"]
G5 = ["log_infra", "log_minor", "log_abd", "infra_ratio", "log_il", "metal"]
GALL = G5 + ["infra_feret_rmax", "infra_grow_eq_rmax", "thor_eq_rmax", "infra_lowfrac", "infra_hu_med", "lum_ref", "v_n2000", "a_top_at_fov"]
P = {}
X = lambda d, cols, extra=(): d[list(extra) + cols].values.astype(float)
P["geo1_lr"] = cv(X(T, G1)); P["geo5_lr"] = cv(X(T, G5)); P["geo_gbm"] = cv(X(T, GALL), "gbm")
P["E3+geo1_lr"] = cv(X(T, G1, ["E3_late"])); P["E3+geo5_lr"] = cv(X(T, G5, ["E3_late"]))
P["E3+geo_gbm"] = cv(X(T, GALL, ["E3_late"]), "gbm")
P["E0+geo5_lr"] = cv(X(T, G5, ["E0"]))
P["age"] = T.Age.values.astype(float); P["age_sex_lr"] = cv(X(T, [], ["Age", "male"]))
P["E3+age_lr"] = cv(X(T, [], ["E3_late", "Age", "male"]))
P["E3+geo5+age_lr"] = cv(X(T, G5, ["E3_late", "Age", "male"]))
for k, v in P.items():
    full("3_stack_trainOOF", k, y, v)
for k in ("E3+geo1_lr", "E3+geo5_lr", "E3+geo_gbm"):
    delta("3_stack_trainOOF", k + " minus E3_late", y, P[k], S3)
# val descriptive: fit on full train, apply to val
for nm, cols, ex, kind in (("E3+geo1_lr", G1, ["E3_late"], "lr"), ("E3+geo5_lr", G5, ["E3_late"], "lr"), ("geo5_lr", G5, [], "lr"),
                           ("E3+geo_gbm", GALL, ["E3_late"], "gbm")):
    pv = fitval(X(T, cols, ex), X(V, cols, ex), kind); V["p_" + nm] = pv
    full("3_stack_val_descriptive", nm, yv, pv)
    delta("3_stack_val_descriptive", nm + " minus E3_late(this recipe)", yv, pv, V.E3_late.values)
full("3_stack_val_descriptive", "val infra_eq_rmax alone", yv, V.infra_eq_rmax.values.astype(float))
for k, v in P.items(): T["p_" + k] = v
ST = P["E3+geo5_lr"]

# ------------------------------------------------------------------ 4. subgroups (positives vs all negatives; negatives subgroup vs all positives)
pos, ng = y == 1, y == 0
qd = T.q_diam_cm.values
meas = T.infra_eq_rmax.values
T["repaired"] = ((T.q_repaired == "yes") | (T.kw_evar == 1)).astype(int)
T["metal_any"] = (T.v_n1600.fillna(0) > 0).astype(int)
psub = {
 "repaired / EVAR (Qwen repaired=yes or EVAR/graft keyword)": T.repaired == 1,
 "not repaired": T.repaired == 0,
 "reported size 3.0-3.4 cm": (qd >= 3.0) & (qd < 3.5), "reported size 3.5-3.9 cm": (qd >= 3.5) & (qd < 4.0),
 "reported size 4.0-4.9 cm": (qd >= 4.0) & (qd < 5.0), "reported size >=5 cm": qd >= 5.0, "size not stated": np.isnan(qd),
 "measured infrarenal eq. diameter < 30 mm": meas < 30, "measured >= 30 mm": meas >= 30,
 "iliac aneurysm keyword": T.kw_iliac_aneur == 1, "thoracic/thoracoabdominal keyword": T.kw_thoracic == 1,
 "supra/juxtarenal keyword": T.kw_suprarenal == 1, "mural thrombus keyword": T.kw_thrombus == 1,
 "dissection keyword": T.kw_dissection == 1, "ectasia keyword in positive": T.kw_ectasia == 1,
 "non-contrast": T.contrast_b == 0, "aorta top at FOV edge": T.a_top_at_fov == 1, "no kidney (renal level fallback)": T.has_kidney == 0,
 "age < 70": T.Age < 70, "age >= 80": T.Age >= 80, "female": T.male == 0,
}
rows_pos = []
for k, m in psub.items():
    m = np.asarray(m, bool)
    mp = m & pos
    if mp.sum() < 3:
        rec("4_pos_subgroup", k, mp.sum(), ng.sum(), np.nan, note="n<3"); continue
    for nm, s in (("E3_late", S3), ("E3+geo5", ST), ("infra_diam", meas.astype(float))):
        keep = mp | ng
        a = auc(y[keep], s[keep]); lo, hi = boot(y[keep], s[keep], nb=1000)
        rec("4_pos_subgroup", f"{k} | {nm}", mp.sum(), ng.sum(), a, lo, hi,
            note=f"frac of pos {mp.sum()/pos.sum():.3f}; cost(E3) {mp.sum()/pos.sum()*(rt[~m[pos]].mean()-rt[m[pos]].mean()) if (~m[pos]).sum() else np.nan:.4f}" if nm == "E3_late" else "")
nsub = {
 "ectasia/borderline negatives (ectasia kw or Qwen diameter 2.5-2.9)": (T.kw_ectasia == 1) | ((qd >= 2.5) & (qd < 3.0)),
 "negatives with measured infra diameter >= 30 mm": meas >= 30,
 "negatives with measured 25-30 mm": (meas >= 25) & (meas < 30),
 "negatives with EVAR/graft keyword or metal >1600 HU near aorta": (T.kw_evar == 1) | (T.v_n1600.fillna(0) > 0),
 "negatives with iliac aneurysm keyword": T.kw_iliac_aneur == 1,
 "negatives with thoracic aneurysm keyword": T.kw_thoracic == 1,
 "negatives category not_mentioned": T.aaa_cat == "not_mentioned", "negatives category no": T.aaa_cat == "no",
 "negatives probably_no": T.aaa_cat == "probably_no",
 "negatives age >= 80": T.Age >= 80, "negatives male >=70": (T.male == 1) & (T.Age >= 70),
}
posr = np.sort(S3[pos]); rn = 1 - np.searchsorted(posr, S3[ng]) / len(posr)
for k, m in nsub.items():
    m = np.asarray(m, bool); mn = m & ng
    if mn.sum() < 3:
        rec("4_neg_subgroup", k, pos.sum(), mn.sum(), np.nan, note="n<3"); continue
    keep = pos | mn
    for nm, s in (("E3_late", S3), ("E3+geo5", ST)):
        a = auc(y[keep], s[keep]); lo, hi = boot(y[keep], s[keep], nb=1000)
        mm_ = m[ng]
        rec("4_neg_subgroup", f"{k} | {nm}", pos.sum(), mn.sum(), a, lo, hi,
            note=f"frac of neg {mn.sum()/ng.sum():.4f}; cost(E3) {mm_.mean()*(rn[~mm_].mean()-rn[mm_].mean()):.4f}" if nm == "E3_late" else "")
# clean-label evaluation: positives vs negatives explicitly 'no' and measured < 27 mm
clean_neg = ng & (T.aaa_cat == "no").values
full("4_clean", "E3: positives vs explicit-'no' negatives", y[pos | clean_neg], S3[pos | clean_neg])
full("4_clean", "E3: unrepaired positives vs all negatives", y[(pos & (T.repaired == 0).values) | ng], S3[(pos & (T.repaired == 0).values) | ng])

# ------------------------------------------------------------------ 5. thrombus / mask
c = T.contrast_b.values == 1
for nm, cols in (("infra_lowfrac (contrast, mask voxels < lumen-80 HU)", "infra_lowfrac"), ("infra_grow_ratio", "infra_grow_ratio")):
    v = T[cols].values.astype(float)
    a = np.nanmedian(v[pos & c]); b = np.nanmedian(v[ng & c])
    rec("5_mask", nm + ": median pos / neg (contrast)", (pos & c).sum(), (ng & c).sum(), a, note=f"neg median {b:.4f}; AUROC {auc(y[c], v[c]):.4f}")
th = (T.kw_thrombus == 1).values
for nm, m in (("thrombus kw", pos & th), ("no thrombus kw", pos & ~th)):
    mm2 = m & ~np.isnan(q)
    if mm2.sum() >= 3:
        d = meas[mm2] - q[mm2]
        rec("5_mask", f"measured - reported (mm), positives {nm}", mm2.sum(), 0, np.median(d), *np.percentile(d, [25, 75]), note="CI cols = IQR")
    lf = T.infra_lowfrac.values[m & c]
    if (m & c).sum() >= 3:
        rec("5_mask", f"infra_lowfrac median, positives {nm} (contrast)", (m & c).sum(), 0, np.nanmedian(lf))

# ------------------------------------------------------------------ 6. age
full("6_age", "age only", y, T.Age.values.astype(float))
b5 = (T.Age // 5).values
def strat_auc(s, yy=y, bins=b5):
    num = den = 0.0
    for bb in np.unique(bins[~np.isnan(bins)]):
        m = bins == bb; p_, n_ = (yy[m] == 1).sum(), (yy[m] == 0).sum()
        if p_ and n_:
            num += auc(yy[m], s[m]) * p_ * n_; den += p_ * n_
    return num / den
for nm, s in (("E0", S0), ("E3_late", S3), ("E3+geo5", ST), ("infra_diam", meas.astype(float))):
    sv = np.nan_to_num(s, nan=np.nanmedian(s))
    bs = []
    P_, N_ = np.where(pos)[0], np.where(ng)[0]
    for _ in range(500):
        i = np.r_[rng.choice(P_, len(P_)), rng.choice(N_, len(N_))]
        bs.append(strat_auc(sv[i], y[i], b5[i]))
    rec("6_age_matched", nm + " (5-yr age-stratified AUROC)", pos.sum(), ng.sum(), strat_auc(sv), *np.percentile(bs, [2.5, 97.5]))
rec("6_age", "spearman(E3, age) in negatives", 0, ng.sum(), spearmanr(S3[ng], T.Age[ng], nan_policy="omit")[0])
rec("6_age", "spearman(infra diam, age) in negatives", 0, ng.sum(), spearmanr(meas[ng], T.Age[ng], nan_policy="omit")[0])
rec("6_age", "spearman(E3, infra diam) in negatives", 0, ng.sum(), spearmanr(S3[ng], meas[ng], nan_policy="omit")[0])
rec("6_age", "median age positives", pos.sum(), 0, T.Age[pos].median(), note=f"negatives {T.Age[ng].median():.1f}")
for lo_ in (60, 70):
    m = (T.Age >= lo_).values
    full("6_age", f"E3 among age >= {lo_}", y[m], S3[m]); full("6_age", f"E3+geo5 among age >= {lo_}", y[m], ST[m])
full("6_age", "male only E3", y[T.male.values == 1], S3[T.male.values == 1])
full("6_age", "female only E3", y[T.male.values == 0], S3[T.male.values == 0])

# ------------------------------------------------------------------ 7. errors at 90% / 95% specificity
def errs(nm, s, spec):
    thr = np.quantile(s[ng], spec)
    fn = pos & (s < thr); fp = ng & (s >= thr)
    rec("7_errors", f"{nm} @ {int(spec*100)}% spec: sensitivity", pos.sum(), ng.sum(), 1 - fn.sum() / pos.sum(), note=f"FN {fn.sum()} FP {fp.sum()}")
    return fn, fp
for spec in (0.90, 0.95, 0.98):
    fn, fp = errs("E3_late", S3, spec); errs("E3+geo5", ST, spec); errs("infra_diam", np.nan_to_num(meas, nan=0), spec)
fn, fp = (pos & (S3 < np.quantile(S3[ng], 0.90))), (ng & (S3 >= np.quantile(S3[ng], 0.90)))
fn95 = pos & (S3 < np.quantile(S3[ng], 0.95))
def fncat(r):
    if r.repaired == 1: return "A_repaired_EVAR"
    if pd.notna(r.infra_eq_rmax) and r.infra_eq_rmax >= 30: return "B_large_on_mask_but_missed"
    if (pd.notna(r.q_diam_cm) and r.q_diam_cm < 3.5) or (pd.notna(r.infra_eq_rmax) and r.infra_eq_rmax < 30 and pd.isna(r.q_diam_cm)): return "C_small_or_borderline"
    if r.kw_iliac_aneur == 1 or r.kw_thoracic == 1 or r.kw_suprarenal == 1: return "D_iliac_thoracic_suprarenal"
    if pd.notna(r.q_diam_cm) and pd.notna(r.infra_eq_rmax) and r.infra_eq_rmax < r.q_diam_cm * 10 - 5: return "E_mask_underestimate"
    return "F_other"
def fpcat(r):
    if r.kw_evar == 1 or (r.v_n1600 or 0) > 0: return "A_graft_or_metal"
    if pd.notna(r.infra_eq_rmax) and r.infra_eq_rmax >= 30: return "B_measured>=30mm(possible label miss)"
    if r.kw_ectasia == 1 or (pd.notna(r.q_diam_cm) and r.q_diam_cm >= 2.5) or (pd.notna(r.infra_eq_rmax) and r.infra_eq_rmax >= 25): return "C_ectasia_borderline_25-30mm"
    if r.kw_iliac_aneur == 1 or r.kw_thoracic == 1: return "D_iliac_or_thoracic_aneurysm"
    if r.Age >= 75: return "E_old_tortuous"
    return "F_other"
T["fn_cat"] = ""; T.loc[fn, "fn_cat"] = T[fn].apply(fncat, axis=1)
T["fn95_cat"] = ""; T.loc[fn95, "fn95_cat"] = T[fn95].apply(fncat, axis=1)
T["fp_cat"] = ""; T.loc[fp, "fp_cat"] = T[fp].apply(fpcat, axis=1)
allp = T[pos].apply(fncat, axis=1).value_counts(normalize=True)
alln = T[ng].apply(fpcat, axis=1).value_counts(normalize=True)
for k, v in T[fn].fn_cat.value_counts().items():
    rec("7_FN90_cat", k, v, 0, v / fn.sum(), note=f"share among all pos {allp.get(k, 0):.3f}")
for k, v in T[fn95].fn95_cat.value_counts().items():
    rec("7_FN95_cat", k, v, 0, v / fn95.sum(), note=f"share among all pos {allp.get(k, 0):.3f}")
for k, v in T[fp].fp_cat.value_counts().items():
    rec("7_FP90_cat", k, 0, v, v / fp.sum(), note=f"share among all neg {alln.get(k, 0):.4f}")
print("FN @90 detail:\n", T[fn][["aaa_cat", "q_diam_cm", "q_repaired", "kw_evar", "kw_iliac_aneur", "kw_thoracic", "kw_thrombus", "infra_eq_rmax", "abd_eq_rmax", "v_n1600", "Age", "rank_E3", "fn_cat"]].round(2).to_string(index=False))
print("FN @95 n", fn95.sum())
# ------------------------------------------------------------------ 8. ceiling
pr = np.sort(rt)
rec("8_ceiling", "E3: mean rank of positives (=AUROC)", pos.sum(), ng.sum(), rt.mean())
rep = (T.repaired == 1).values[pos]
rec("8_ceiling", "E3: AUROC if repaired positives were removed", (~rep).sum(), ng.sum(), rt[~rep].mean())
rec("8_ceiling", "E3+geo5: AUROC on positives vs explicit-'no' negatives", pos.sum(), clean_neg.sum(), auc(y[pos | clean_neg], ST[pos | clean_neg]))
keep = cols = ["study_id", "fold", "y_aaa", "aaa_cat", "q_diam_cm", "q_repaired", "repaired", "kw_ectasia", "kw_evar", "kw_iliac_aneur", "kw_thoracic",
        "kw_suprarenal", "kw_thrombus", "Age", "Gender", "contrast_b", "E0", "E3_late", "rank_E3"] + [c for c in GEO if c in T] + \
       ["infra_lowfrac", "infra_grow_ratio", "v_n1600", "v_n2000", "a_top_at_fov", "p_E3+geo5_lr", "p_E3+geo_gbm", "p_geo5_lr", "fn_cat", "fn95_cat", "fp_cat"]
T[keep].to_csv(D + "per_case_train_oof.csv", index=False)
V[["study_id", "y_aaa", "expB_E0", "expB_E3", "E3_late", "infra_eq_rmax", "p_E3+geo1_lr", "p_E3+geo5_lr", "p_geo5_lr", "p_E3+geo_gbm"]].to_csv(D + "scores_val_descriptive_stack.csv", index=False)
pd.DataFrame(R).to_csv(D + "results_train_oof.csv", index=False)
print("ANALYZE_DONE")
