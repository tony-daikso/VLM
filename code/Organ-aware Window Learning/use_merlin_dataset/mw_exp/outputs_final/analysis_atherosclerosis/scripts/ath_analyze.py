"""Atherosclerosis error analysis on TRAIN 5-fold OOF (val descriptive only; test never read)."""
import os
try: os.nice(19)
except OSError: pass
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
D = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_atherosclerosis/"
lab = pd.read_csv(D + "labels_trval.csv")
hu = pd.concat([pd.read_csv(D + "hu_a6.csv"), pd.read_csv(D + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "slicethickness", "kvp"]]
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
T = df[df.split == "train"].merge(pd.read_csv(D + "oof_train.csv"), on="study_id"); T = T[T.y_ath.notna()].reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(D + "scores_val_descriptive.csv"), on="study_id"); V = V[V.y_ath.notna()].reset_index(drop=True)
y = T.y_ath.values.astype(int); folds = T.fold.values
out = []
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
def rec(sec, item, n1, n0, val, note=""):
    out.append(dict(section=sec, item=item, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
    print(f"[{sec}] {item}: {val:.4f} (pos {n1}, neg {n0}) {note}", flush=True)
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"] if c in T]
for m in MODES: rec("1_overall_oof", m, y.sum(), (1 - y).sum(), auc(y, T[m]))
print("val descriptive:", {m: round(auc(V.y_ath.astype(int), V[m]), 4) for m in MODES})
# ---------- features
for d in (T, V):
    d["phase2"] = np.where(d.contrast == False, "non_contrast", d.phase.fillna("unknown"))
    d["lum_bin"] = pd.cut(d.lum_p50, [-1e9, 100, 200, 300, 1e9], labels=["lumen<100", "100-200", "200-300", ">300"]).astype(str)
    d["male"] = (d.Gender == "Male").astype(float)
    d["il_present"] = (d.il_vox.fillna(0) >= 20).astype(int)
CNT = [c for c in T.columns if c.startswith(("a_", "il_", "v_")) and c not in ("a_vox", "il_vox", "il_present")]
for c in CNT: rec("2_hu_single", c, y.sum(), (1 - y).sum(), auc(y, T[c].fillna(0)))
rec("2_hu_single", "lumen p50 (contrast proxy)", y.sum(), (1 - y).sum(), auc(y, T.lum_p50))
rec("2_hu_single", "a_vox (aorta size)", y.sum(), (1 - y).sum(), auc(y, T.a_vox))
CALC = "v_massA"
T["calc"] = np.log1p(T[CALC].fillna(0)); V["calc"] = np.log1p(V[CALC].fillna(0))
def X(d, cols, extra=()):
    x = pd.DataFrame(index=d.index)
    for c in cols:
        v = d[c].astype(float)
        x[c] = np.log1p(v.clip(lower=0)) if c.endswith(("_vox", "massA")) or "_n" in c else v
    x = x.fillna(0)
    for e in extra: x[e] = d[e].astype(float).fillna(T[e].median())
    return x
HUc = CNT + ["a_vox", "il_vox", "lum_p50", "lum_p90"]
AOc = [c for c in CNT if c.startswith("a_")] + ["a_vox", "lum_p50", "lum_p90"]
def cv(xT, model="lr"):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = folds != k, folds == k
        m = (make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")) if model == "lr"
             else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))
        m.fit(xT[tr], y[tr]); p[te] = m.decision_function(xT[te]) if model == "lr" else m.predict_proba(xT[te])[:, 1]
    return p
P = {}
P["hu_lr"] = cv(X(T, HUc)); P["hu_gbm"] = cv(X(T, HUc), "gbm"); P["hu_aorta_only_lr"] = cv(X(T, AOc))
P["age"] = T.Age.values; P["age_sex_lr"] = cv(X(T, [], ["Age", "male"]))
P["hu+age_gbm"] = cv(X(T, HUc, ["Age", "male"]), "gbm")
P["E3+hu_lr"] = cv(X(T, HUc, ["E3_late"])); P["E3+hu_gbm"] = cv(X(T, HUc, ["E3_late"]), "gbm")
P["E3+age_lr"] = cv(X(T, [], ["E3_late", "Age", "male"]))
P["E3+hu+age_gbm"] = cv(X(T, HUc, ["E3_late", "Age", "male"]), "gbm")
wl = [m for m in MODES if m.startswith("E1")] + ["E0"]
P["windows_stack_lr"] = cv(X(T, [], wl)); P["windows+hu_gbm"] = cv(X(T, HUc, wl), "gbm")
for k, v in P.items(): rec("3_models_oof", k, y.sum(), (1 - y).sum(), auc(y, v))
for k, v in P.items(): T["p_" + k] = v
# val descriptive (fit on all train)
def fitall(cols, extra, model):
    m = (make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")) if model == "lr"
         else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))
    m.fit(X(T, cols, extra), y); xv = X(V, cols, extra)
    return m.decision_function(xv) if model == "lr" else m.predict_proba(xv)[:, 1]
yv = V.y_ath.values.astype(int)
for nm, cols, ex, mo in (("hu_gbm", HUc, (), "gbm"), ("E3+hu_lr", HUc, ("E3_late",), "lr"), ("E3+hu_gbm", HUc, ("E3_late",), "gbm")):
    rec("3b_val_descriptive", nm, yv.sum(), (1 - yv).sum(), auc(yv, fitall(cols, ex, mo)))
rec("3b_val_descriptive", "E3_late (this recipe)", yv.sum(), (1 - yv).sum(), auc(yv, V.E3_late))
rec("3b_val_descriptive", "E0 (this recipe)", yv.sum(), (1 - yv).sum(), auc(yv, V.E0))
S = T.E3_late.values; H = T.p_hu_gbm.values; ST = T["p_E3+hu_gbm"].values
def sub(sec, name, mask):
    mask = np.asarray(mask, bool)
    if (y[mask] == 1).sum() < 20 or (y[mask] == 0).sum() < 20: return
    rec(sec, name, (y[mask] == 1).sum(), (y[mask] == 0).sum(), auc(y[mask], S[mask]),
        f"E0={auc(y[mask], T.E0.values[mask]):.3f} HU={auc(y[mask], H[mask]):.3f} E3+HU={auc(y[mask], ST[mask]):.3f} calc={auc(y[mask], T.calc.values[mask]):.3f} prev={y[mask].mean():.3f}")
# ---------- phase
for g in sorted(T.phase2.unique()): sub("4_phase", g, T.phase2 == g)
for g in ["lumen<100", "100-200", "200-300", ">300"]: sub("4_lumen", g, T.lum_bin == g)
# calcium detectability by lumen bin
for g in ["lumen<100", "100-200", "200-300", ">300"]:
    m = (T.lum_bin == g).values
    print(f"  calc>0 rate {g}: pos {(T.calc[m & (y==1)] > 0).mean():.3f} neg {(T.calc[m & (y==0)] > 0).mean():.3f}")
# ---------- age
rec("5_age", "age only AUROC", y.sum(), (1 - y).sum(), auc(y, T.Age))
T["agebin"] = pd.cut(T.Age, [0, 50, 60, 70, 80, 200], labels=["<50", "50-60", "60-70", "70-80", ">=80"]).astype(str)
for g in ["<50", "50-60", "60-70", "70-80", ">=80"]: sub("5_age_strata", g, T.agebin == g)
b5 = (T.Age // 5).values
def strat_auc(s):
    num = den = 0
    for b in np.unique(b5[~np.isnan(b5)]):
        m = b5 == b; n1, n0 = (y[m] == 1).sum(), (y[m] == 0).sum()
        if n1 and n0: a = auc(y[m], s[m]); num += a * n1 * n0; den += n1 * n0
    return num / den
for nm, s in (("E0", T.E0.values), ("E3_late", S), ("hu_gbm", H), ("E3+hu_gbm", ST), ("calc", T.calc.values)):
    rec("5_age_matched", nm + " (5-yr age-stratified AUROC)", y.sum(), (1 - y).sum(), strat_auc(s))
rec("5_age", "spearman(E3, age) in negatives", 0, (y == 0).sum(), spearmanr(S[y == 0], T.Age[y == 0], nan_policy="omit")[0])
rec("5_age", "spearman(calc, age) in negatives", 0, (y == 0).sum(), spearmanr(T.calc[y == 0], T.Age[y == 0], nan_policy="omit")[0])
# ---------- severity / text
for g in ["mild", "moderate", "severe", "not_stated"]:
    sub("6_grade", "grade=" + g, (T.grade == g) | (y == 0))
    m = (T.grade == g) & (y == 1); print(f"  grade {g}: n={m.sum()} calc median {T.calc[m].median():.2f} calc>0 {(T.calc[m]>0).mean():.3f}")
for k in ["kw_aortoiliac", "kw_vasc_calc", "kw_coronary", "kw_noncalc", "g_mild", "g_severe"]:
    sub("6_pos_text", k + "=1 (pos) vs all neg", ((T[k] == 1) & (y == 1)) | (y == 0))
    sub("6_pos_text", k + "=0 (pos) vs all neg", ((T[k] == 0) & (y == 1)) | (y == 0))
print("  neg calc median", T.calc[y == 0].median(), "pos", T.calc[y == 1].median())
# ---------- label noise
negs = y == 0
q50, q25 = np.percentile(T.calc[y == 1], [50, 25])
for nm, thr in (("pos median", q50), ("pos p25", q25)):
    rec("7_label_noise", f"frac neg with calc >= {nm} ({thr:.2f})", 0, negs.sum(), (T.calc[negs] >= thr).mean())
    rec("7_label_noise", f"frac not_mentioned neg with calc >= {nm}", 0, (T.ath_cat == "not_mentioned").sum(), (T.calc[T.ath_cat == "not_mentioned"] >= thr).mean())
    rec("7_label_noise", f"frac explicit-no neg with calc >= {nm}", 0, (T.ath_cat == "no").sum(), (T.calc[T.ath_cat == "no"] >= thr).mean())
rec("7_label_noise", "frac pos with calc==0 (no voxel > adaptive thr)", y.sum(), 0, (T.calc[y == 1] == 0).mean())
txtneg = negs & ((T.kw_vasc_calc == 1) | (T.kw_athero == 1) | (T.kw_calc_plaque == 1)).values & (T.ath_cat != "no").values
rec("7_label_noise", "frac neg whose report mentions vascular calcification/plaque (not 'no')", 0, negs.sum(), txtneg.mean())
rec("7_label_noise", "calc median: text-flagged neg", 0, txtneg.sum(), T.calc[txtneg].median())
rec("7_label_noise", "E3 AUROC pos vs text-flagged neg", y.sum(), txtneg.sum(), auc(y[(y == 1) | txtneg], S[(y == 1) | txtneg]))
m = ~txtneg; rec("7_label_noise", "E3 AUROC excluding text-flagged neg", y.sum(), (negs & m).sum(), auc(y[m], S[m]))
hi = negs & (T.calc.values >= q50)
m = ~hi; rec("7_label_noise", "E3 AUROC excluding neg with calc>=pos median", y.sum(), (negs & m).sum(), auc(y[m], S[m]))
m = ~(hi | txtneg); rec("7_label_noise", "E3 AUROC excluding both", y.sum(), (negs & m).sum(), auc(y[m], S[m]))
for g in ["no", "not_mentioned"]: sub("7_neg_type", "pos vs neg=" + g, (y == 1) | (T.ath_cat == g))
# ---------- windows
lg = T[wl].values; c = np.corrcoef(lg.T)
rec("8_windows", "mean pairwise corr of window logits (OOF)", y.sum(), (1 - y).sum(), c[np.triu_indices(len(wl), 1)].mean(), str(wl))
for m_ in wl + ["E3_late"]:
    rec("8_windows", f"spearman({m_}, calc) all", 0, len(T), spearmanr(T[m_], T.calc)[0])
    sub("8_windows_calc0", f"{m_}: calc==0 pos vs calc==0 neg", T.calc == 0)
for m_ in wl + ["E3_late"]:
    mm = (T.calc == 0).values
    rec("8_windows_calc0_auc", m_, (y[mm] == 1).sum(), (y[mm] == 0).sum(), auc(y[mm], T[m_].values[mm]))
    mm = (T.calc > 0).values
    rec("8_windows_calcpos_auc", m_, (y[mm] == 1).sum(), (y[mm] == 0).sum(), auc(y[mm], T[m_].values[mm]))
# ---------- coverage
sub("9_coverage", "iliac present (>=20 vox)", T.il_present == 1); sub("9_coverage", "iliac absent", T.il_present == 0)
T["a_vox_q"] = pd.qcut(T.a_vox, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
for g in ["Q1", "Q2", "Q3", "Q4"]: sub("9_coverage", "aorta vox " + g, T.a_vox_q == g)
sub("9_coverage", "y_aaa=1 excluded", T.y_aaa != 1)
# ---------- errors at 90% specificity
def errs(name, s):
    thr = np.percentile(s[y == 0], 90); pred = s >= thr
    fn = (y == 1) & ~pred; fp = (y == 0) & pred
    rec("10_spec90", f"{name}: sensitivity", y.sum(), (1 - y).sum(), pred[y == 1].mean(), f"FN={fn.sum()} FP={fp.sum()}")
    return fn, fp, thr
fn, fp, thr = errs("E3_late", S); errs("E3+hu_gbm", ST); errs("hu_gbm", H); errs("age", T.Age.values)
qneg50 = np.percentile(T.calc[y == 0], 50)
def fncat(r):
    if r.calc == 0: return "A_no_calcium_detected"
    if r.grade == "mild" or r.g_mild == 1: return "B_mild_with_calcium"
    if r.calc < q25: return "C_low_calcium"
    if r.Age < 60: return "D_young_with_calcium"
    return "E_calcified_other"
def fpcat(r):
    if r.calc >= q50 or (r.ath_cat != "no" and (r.kw_vasc_calc or r.kw_athero or r.kw_calc_plaque)): return "A_heavy_calcium_or_text_mention(likely_unlabeled)"
    if r.y_aaa == 1 or r.kw_stent_graft and r.calc > 0: return "B_aneurysm_or_graft"
    if r.calc >= q25: return "C_moderate_calcium"
    if r.Age >= 70: return "D_old_low_calcium"
    return "E_other"
T["fn_cat"] = ""; T["fp_cat"] = ""
T.loc[fn, "fn_cat"] = T[fn].apply(fncat, axis=1); T.loc[fp, "fp_cat"] = T[fp].apply(fpcat, axis=1)
allpos = T[y == 1].apply(fncat, axis=1).value_counts(normalize=True); allneg = T[y == 0].apply(fpcat, axis=1).value_counts(normalize=True)
for k, v in T.fn_cat[fn].value_counts(normalize=True).items(): rec("10_FN_cat", k, fn.sum(), 0, v, f"base rate in all pos={allpos.get(k, 0):.3f}")
for k, v in T.fp_cat[fp].value_counts(normalize=True).items(): rec("10_FP_cat", k, 0, fp.sum(), v, f"base rate in all neg={allneg.get(k, 0):.3f}")
print("FN median age", T.Age[fn].median(), "TP median age", T.Age[(y == 1) & ~fn].median(), "FP median age", T.Age[fp].median(), "TN", T.Age[(y == 0) & ~fp].median())
pd.DataFrame(out).to_csv(D + "results_train_oof.csv", index=False)
keep = ["study_id", "fold", "y_ath", "ath_cat", "grade", "aortoiliac", "Age", "Gender", "phase2", "lum_p50", "a_vox", "il_vox", "calc", "v_nA", "v_n300", "v_zfracA",
        "E0", "E3_late"] + [m for m in MODES if m.startswith("E1")] + ["p_hu_gbm", "p_E3+hu_gbm", "fn_cat", "fp_cat"] + [c for c in T.columns if c.startswith(("kw_", "g_"))]
T[keep].to_csv(D + "per_case_train_oof.csv", index=False)
print("ANALYZE_DONE")
