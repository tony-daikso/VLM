"""Appendicitis error analysis on TRAIN 5-fold OOF (val descriptive only; test never loaded)."""
import numpy as np, pandas as pd, json, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_appendicitis/"
rng = np.random.default_rng(0)
lab = pd.read_csv(A + "labels_trval.csv")
hu = pd.concat([pd.read_csv(A + "hu_a6.csv"), pd.read_csv(A + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase", "manufacturer"]]
oof = pd.read_csv(A + "oof_train.csv")
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
T = df[df.split == "train"].merge(oof, on="study_id")
T = T[T.y_ap.notna()].reset_index(drop=True)
y = T.y_ap.values.astype(int)
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
def ci(yy, s, B=1000):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s); yy, s = yy[m], s[m]
    p, n = np.where(yy == 1)[0], np.where(yy == 0)[0]
    if len(p) < 3 or len(n) < 3: return (np.nan, np.nan)
    v = [roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[s[rng.choice(p, len(p))], s[rng.choice(n, len(n))]]) for _ in range(B)]
    return tuple(np.percentile(v, [2.5, 97.5]))
def dci(yy, s1, s2, B=1000):
    """paired stratified bootstrap of AUROC(s2)-AUROC(s1)"""
    yy = np.asarray(yy); p, n = np.where(yy == 1)[0], np.where(yy == 0)[0]; v = []
    for _ in range(B):
        i = np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))]
        v.append(roc_auc_score(yy[i], s2[i]) - roc_auc_score(yy[i], s1[i]))
    return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
out = []
def rec(sec, name, yy, s, note="", withci=True):
    a = auc(yy, s); lo, hi = ci(yy, s) if withci else (np.nan, np.nan)
    out.append(dict(section=sec, item=name, n_pos=int(np.sum(yy)), n_neg=int(len(yy) - np.sum(yy)), auroc=round(a, 4),
                    ci_lo=round(lo, 3), ci_hi=round(hi, 3), note=note))
def recv(sec, name, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=np.nan, n_neg=np.nan, auroc=round(float(val), 4), ci_lo=np.nan, ci_hi=np.nan, note=note))
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "E3_late_colon_sb", "E3_late_sb"] if c in T]
for m in MODES: rec("1_overall_oof", m, y, T[m].values)
S = T["E3_late"].values
for m in MODES:
    if m != "E0":
        d, lo, hi = dci(y, T.E0.values, T[m].values); recv("1_delta_vs_E0", m, d, f"95%CI [{lo:.3f},{hi:.3f}]")
# ---------- positive subgroups (vs ALL negatives)
neg = y == 0
def pos_sub(sec, name, pm):
    pm = np.asarray(pm, bool) & (y == 1); mm = pm | neg
    rec(sec, name, y[mm], S[mm], f"frac_of_pos={pm.sum()/y.sum():.3f}; E0={auc(y[mm], T.E0.values[mm]):.3f}")
pos_sub("2_pos_sub", "all positives", y == 1)
pos_sub("2_pos_sub", "category=yes", T.ap_cat == "yes"); pos_sub("2_pos_sub", "category=probably_yes", T.ap_cat == "probably_yes")
for v in ["yes", "no", "not_stated"]:
    pos_sub("2_pos_sub", f"perforated={v}", T.ap_perf == v); pos_sub("2_pos_sub", f"appendicolith={v}", T.ap_colith == v)
pos_sub("2_pos_sub", "perforated or abscess (perf=yes or kw abscess)", (T.ap_perf == "yes"))
for k in ["kw_tip", "kw_followup", "kw_postop", "kw_history", "kw_appendectomy", "kw_notvis", "kw_mucocele", "kw_negphrase", "kw_retrocecal", "kw_rlq", "kw_cecal"]:
    pos_sub("2_pos_sub_kw", k + "=1", T[k] == 1)
pos_sub("2_pos_sub", "diameter >=13mm", T.ap_mm >= 13); pos_sub("2_pos_sub", "diameter <13mm", T.ap_mm < 13); pos_sub("2_pos_sub", "diameter not stated", T.ap_mm.isna())
pos_sub("2_pos_sub", "early/mild/uncertain wording (probably_yes or early|mild in quote)", (T.ap_cat == "probably_yes") | (T.kw_tip == 1))
pos_sub("2_pos_sub", "history/follow-up/post-op (kw_followup|kw_postop)", (T.kw_followup == 1) | (T.kw_postop == 1))
pos_sub("2_pos_sub", "cecal proxy found", T.rlq_r15_n.notna()); pos_sub("2_pos_sub", "cecal proxy missing", T.rlq_r15_n.isna())
# ---------- negative composition (all positives vs negative subsets)
pos = y == 1
def neg_sub(sec, name, nm):
    nm = np.asarray(nm, bool) & neg; mm = pos | nm
    rec(sec, name, y[mm], S[mm], f"frac_of_neg={nm.sum()/neg.sum():.3f}; mean E3 logit neg-sub={S[nm].mean():.2f} (all neg {S[neg].mean():.2f}, pos {S[pos].mean():.2f})")
neg_sub("3_neg_sub", "neg: appendix explicitly normal (kw_negphrase)", T.kw_negphrase == 1)
neg_sub("3_neg_sub", "neg: category not_mentioned", T.ap_cat == "not_mentioned")
neg_sub("3_neg_sub", "neg: category probably_no", T.ap_cat == "probably_no")
neg_sub("3_neg_sub", "neg: appendectomy", T.kw_appendectomy == 1); neg_sub("3_neg_sub", "neg: appendix not visualized", T.kw_notvis == 1)
neg_sub("3_neg_sub", "neg: RLQ/cecal/ileal inflammation words", (T.kw_cecal == 1) & (T.kw_stranding == 1))
for k in ["y_bowel_obstruction", "y_ascites", "y_free_air"]:
    neg_sub("3_neg_sub", f"neg: {k}=1", T[k] == 1)
    rec("3_cooccur", f"{k} rate pos={T[k][pos].mean():.3f} neg={T[k][neg].mean():.3f}; AUROC of label as score", y, T[k].fillna(0).values, withci=False)
# ---------- confounders: age, phase, sex
T["phase2"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
T["agebin"] = pd.cut(T.Age, [0, 30, 45, 60, 75, 120], labels=["<30", "30-45", "45-60", "60-75", ">=75"])
rec("4_age", "age-only AUROC (younger=higher score)", y, -T.Age.values)
recv("4_age", "median age pos / neg", T.Age[pos].median(), f"neg {T.Age[neg].median()}")
recv("4_age", "spearman(E3 logit, age) in negatives", pd.Series(S[neg]).corr(pd.Series(T.Age.values[neg]), method="spearman"))
recv("4_age", "spearman(E3 logit, age) in positives", pd.Series(S[pos]).corr(pd.Series(T.Age.values[pos]), method="spearman"))
for v in ["<30", "30-45", "45-60", "60-75", ">=75"]:
    mm = (T.agebin == v).values
    if y[mm].sum() >= 10: rec("4_age_strata", f"age {v}", y[mm], S[mm], f"pos rate={y[mm].mean():.3f}; E0={auc(y[mm], T.E0.values[mm]):.3f}")
# age-matched (5 negatives per positive within +-2 y, same sex), repeated 20x
def matched(score, reps=20, k=5):
    vals = []; Ag = T.Age.values; G = T.Gender.values
    ni = np.where(neg & ~np.isnan(Ag))[0]
    for r in range(reps):
        rr = np.random.default_rng(r); sel = []
        for i in np.where(pos & ~np.isnan(Ag))[0]:
            c = ni[(np.abs(Ag[ni] - Ag[i]) <= 2) & (G[ni] == G[i])]
            if len(c): sel += list(rr.choice(c, min(k, len(c)), replace=False)) + [i]
        sel = np.array(sel); vals.append(auc(y[sel], score[sel]))
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
for m in ["E0", "E3_late"]:
    a, lo, hi = matched(T[m].values); recv("4_age", f"age+sex-matched AUROC {m} (1:5, +-2y, 20 reps)", a, f"rep range [{lo:.3f},{hi:.3f}]")
a, _, _ = matched(-T.Age.values.astype(float)); recv("4_age", "age-only AUROC within matched set (sanity ~0.5)", a)
for col in ["phase2", "Gender", "manufacturer"]:
    for v in T[col].dropna().unique():
        mm = (T[col] == v).values
        if y[mm].sum() >= 15: rec("4_strata", f"{col}={v}", y[mm], S[mm], f"pos={y[mm].sum()}")
# ---------- HU features
HUF = [c for c in T.columns if c.startswith(("rlq_", "llq_", "dlr_", "dglob_", "glob_")) and c not in ("rlq_zmin",)] + ["colon_vox", "sb_vox"]
for f in HUF:
    a = auc(y, T[f].values)
    if not np.isnan(a): out.append(dict(section="5_single_feature", item=f, n_pos=int(y.sum()), n_neg=int(neg.sum()),
        auroc=round(max(a, 1 - a), 4), ci_lo=np.nan, ci_hi=np.nan, note=f"direction={'+' if a >= .5 else '-'}; median pos={T[f][pos].median():.3f} neg={T[f][neg].median():.3f}; missing={T[f].isna().mean():.3f}"))
STK = [c for c in HUF if c.startswith(("rlq_r", "dlr_", "dglob_", "rlq_wall", "rlq_lumen", "rlq_z", "rlq_seed"))] + ["glob_fat_med", "glob_fluid_frac"]
X = T[STK].copy()
for c in STK:
    if c.endswith(("_n", "_vox", "hi_small_n", "hi_big_vox")) and not c.startswith("dlr"): X[c] = np.log1p(X[c].clip(lower=0))
X = X.fillna(X.median()); X["rlq_missing"] = T.rlq_r15_n.isna().astype(int)
def oof_lr(D, C=0.1):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = T.fold != k, T.fold == k
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        mdl.fit(D[tr], y[tr]); p[te] = mdl.decision_function(D[te])
    return p
age = pd.DataFrame({"age": T.Age.fillna(T.Age.median()), "female": (T.Gender == "Female").astype(float)})
W5 = T[["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]]
T["p_hu"] = oof_lr(X); T["p_age"] = oof_lr(age)
T["p_e3_hu"] = oof_lr(pd.concat([X, T[["E3_late"]]], axis=1))
T["p_e0_hu"] = oof_lr(pd.concat([X, T[["E0"]]], axis=1))
T["p_e3_age"] = oof_lr(pd.concat([T[["E3_late"]], age], axis=1))
T["p_e3_hu_age"] = oof_lr(pd.concat([X, T[["E3_late"]], age], axis=1))
T["p_win"] = oof_lr(W5); T["p_win_hu"] = oof_lr(pd.concat([W5, X], axis=1))
T["p_e3_e3cs_hu"] = oof_lr(pd.concat([X, T[["E3_late", "E3_late_colon_sb"]]], axis=1)) if "E3_late_colon_sb" in T else np.nan
for c in ["p_hu", "p_age", "p_e3_age", "p_e0_hu", "p_e3_hu", "p_e3_hu_age", "p_win", "p_win_hu", "p_e3_e3cs_hu"]:
    if c in T and T[c].notna().all():
        d, lo, hi = dci(y, S, T[c].values); rec("6_stack_oof", c, y, T[c].values, f"delta vs E3_late {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(pd.concat([X, T[["E3_late"]]], axis=1), y)
coef = pd.Series(mdl[-1].coef_[0], index=list(X.columns) + ["E3_late"]).sort_values(key=np.abs, ascending=False)
recv("6_stack_coef", "top coefs (std units)", 0, json.dumps(coef.head(12).round(3).to_dict()))
# stack gain by subgroup
for nm, pm in [("perforated=yes", T.ap_perf == "yes"), ("perforated!=yes", T.ap_perf != "yes"), ("appendicolith=yes", T.ap_colith == "yes"), ("probably_yes", T.ap_cat == "probably_yes")]:
    pm = np.asarray(pm, bool) & pos; mm = pm | neg
    recv("6_stack_by_sub", nm, auc(y[mm], T.p_e3_hu.values[mm]), f"E3 {auc(y[mm], S[mm]):.3f} -> E3+HU {auc(y[mm], T.p_e3_hu.values[mm]):.3f}; n_pos={pm.sum()}")
# ---------- errors at 90% spec (E3_late OOF)
thr = np.quantile(S[neg], 0.90); pred = S >= thr
fn = pos & ~pred; fp = neg & pred
recv("7_err", "sensitivity@90%spec E3_late", (pos & pred).sum() / pos.sum(), f"FN={fn.sum()} FP={fp.sum()}")
for c in ["E0", "p_e3_hu", "p_win_hu"]:
    s = T[c].values; recv("7_err", f"sensitivity@90%spec {c}", (s[pos] >= np.quantile(s[neg], .9)).mean())
def cat_fn(r):
    if r.kw_appendectomy or r.kw_negphrase: return "A 標註矛盾（同時提到切除/正常闌尾）"
    if r.kw_followup or r.kw_postop: return "B 治療中/追蹤/術後（引流、抗生素、消退中）"
    if r.ap_cat == "probably_yes": return "C 報告用語不確定（probably）"
    if pd.isna(r.rlq_r15_n): return "D 盲腸定位失敗"
    if r.ap_perf == "yes": return "E 穿孔/膿瘍（訊號在腔外）"
    if r.kw_tip or (r.ap_mm < 10): return "F 早期/尖端/管徑小"
    return "G 典型陽性但漏判"
def cat_fp(r):
    if r.ap_cat == "probably_no": return "a 報告傾向否定（probably_no）"
    if r.kw_appendectomy or r.kw_notvis: return "b 闌尾已切除/未見"
    if r.y_bowel_obstruction == 1 or r.y_free_air == 1 or r.y_ascites == 1: return "c 其他急性腸道/腹膜表現（阻塞、游離氣、腹水）"
    if r.kw_cecal and r.kw_stranding: return "d 盲腸/迴腸末端發炎（模仿者）"
    if r.kw_rlq: return "e 報告提及右下腹其他病灶"
    return "f 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
bp = T[pos].apply(cat_fn, axis=1); bn = T[neg].apply(cat_fp, axis=1)
for k, v in T.fn_cat[fn].value_counts().items():
    recv("7_err_FN", k, v / fn.sum(), f"n={v} of FN {fn.sum()}; base rate in pos={(bp==k).mean():.3f}; miss rate in category={v/(bp==k).sum():.2f}")
for k, v in T.fp_cat[fp].value_counts().items():
    recv("7_err_FP", k, v / fp.sum(), f"n={v} of FP {fp.sum()}; base rate in neg={(bn==k).mean():.3f}")
T["agegrp"] = np.where(T.Age < 45, "<45", ">=45")
recv("7_err", "FP: frac age<45", (T.Age[fp] < 45).mean(), f"all neg {(T.Age[neg] < 45).mean():.3f}")
recv("7_err", "FN: frac age>=60", (T.Age[fn] >= 60).mean(), f"all pos {(T.Age[pos] >= 60).mean():.3f}")
# ---------- ceiling
susp = (pos & ((T.kw_appendectomy == 1) | (T.kw_negphrase == 1) | (T.kw_followup == 1) | (T.kw_postop == 1) | (T.ap_cat == "probably_yes")).values) | \
       (neg & (T.ap_cat == "probably_no").values)
mm = ~susp
rec("8_ceiling", "E3_late after removing suspicious labels", y[mm], S[mm], f"removed pos={(susp&pos).sum()} neg={(susp&neg).sum()}")
rec("8_ceiling", "E3+HU after removing suspicious labels", y[mm], T.p_e3_hu.values[mm])
mm2 = mm & ~(pos & (T.ap_perf == "yes").values)
rec("8_ceiling", "E3_late, clean, non-perforated only", y[mm2], S[mm2])
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
T[["study_id", "fold", "y_ap", "ap_cat", "ap_perf", "ap_colith", "ap_mm", "Age", "phase2", "rlq_r15_strand_frac", "rlq_r15_fat_med", "dlr_r15_fat_med",
   "E0", "E3_late", "p_hu", "p_e3_hu", "fn_cat", "fp_cat"]].to_csv(A + "per_case_train_oof.csv", index=False)
# ---------- VAL descriptive (no tuning)
V = df[df.split == "val"].copy(); V = V[V.y_ap.notna()]
for nm in ["E0", "E3_late"]:
    z = np.load(R + f"results/expB/{nm}.npz", allow_pickle=True); fi = list(z["findings"]).index("appendicitis")
    pv = pd.Series(z["prob"].mean(0)[:, fi], index=z["ids"]); spl = pd.Series(z["split"], index=z["ids"])
    pv = pv[spl == "val"]; V["expB_" + nm] = V.study_id.map(pv)
sv = pd.read_csv(A + "scores_val_descriptive.csv"); V = V.merge(sv, on="study_id", how="left")
yv = V.y_ap.values.astype(int)
XV = V[STK].copy()
for c in STK:
    if c.endswith(("_n", "_vox", "hi_small_n", "hi_big_vox")) and not c.startswith("dlr"): XV[c] = np.log1p(XV[c].clip(lower=0))
XV = XV.fillna(T[STK].median() if False else X.median()); XV["rlq_missing"] = V.rlq_r15_n.isna().astype(int)
XT = pd.concat([X, T[["E3_late"]]], axis=1); mfull = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(XT, y)
V["stack_e3_hu"] = mfull.decision_function(pd.concat([XV, V[["E3_late"]]], axis=1))
mhu = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(X, y); V["hu_only"] = mhu.decision_function(XV)
vrows = []
for nm in ["expB_E0", "expB_E3_late", "E0", "E3_late", "hu_only", "stack_e3_hu"]:
    s = V[nm].values; lo, hi = ci(yv, s)
    vrows.append(dict(model=nm, subset="all", n_pos=yv.sum(), auc=round(auc(yv, s), 4), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
    for sub, pm in [("perforated=yes", V.ap_perf == "yes"), ("perforated!=yes", V.ap_perf != "yes")]:
        mm = (np.asarray(pm) & (yv == 1)) | (yv == 0)
        vrows.append(dict(model=nm, subset=sub, n_pos=int(yv[mm].sum()), auc=round(auc(yv[mm], s[mm]), 4)))
vrows.append(dict(model="age_only", subset="all", n_pos=yv.sum(), auc=round(auc(yv, -V.Age.values), 4)))
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 160); pd.set_option("display.max_rows", 400)
print(res.to_string()); print(pd.DataFrame(vrows).to_string())
