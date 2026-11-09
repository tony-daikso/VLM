"""Ascites error analysis on TRAIN 5-fold OOF (val descriptive only). Inputs: labels_trval.csv, oof_train.csv,
scores_val_descriptive.csv, hu_a6.csv + hu_ct.csv, metadata. Outputs: results_train_oof.csv, per_case_train_oof.csv,
val_descriptive.csv, printed summary."""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import os
A = os.environ.get("ADIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_ascites/")
lab = pd.read_csv(A + "labels_trval.csv")
hu = pd.concat([pd.read_csv(A + f) for f in ("hu_ct.csv", "hu_ct_b.csv") if os.path.exists(A + f)]).drop_duplicates("study_id")   # random train/val subset (CT-rate host), see report
meta = pd.read_csv(A + "metadata_trval.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase"]]
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
oof = pd.read_csv(A + "oof_train.csv")
T = df[df.split == "train"].merge(oof, on="study_id"); T = T[T.y_asc.notna()].reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(A + "scores_val_descriptive.csv"), on="study_id"); V = V[V.y_asc.notna()].reset_index(drop=True)
y = T.y_asc.values.astype(int); S = T.E3_late.values
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "E3_late_all12", "E3_late_dropamb", "E3_late_anyfluid"] if c in T]
out = []
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
def rec(sec, item, n1, n0, val, note=""):
    out.append(dict(section=sec, item=item, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
    print(f"[{sec}] {item}: {val:.4f} (n+={n1}, n-={n0}) {note}" if val == val else f"[{sec}] {item}: nan {note}")
amb = ((T.asc_cat == "probably_no") & T.amount.isin(["trace", "small"])).values
for m in MODES:
    rec("1_overall_oof", m, y.sum(), (1 - y).sum(), auc(y, T[m]), f"excl. ambiguous neg={auc(y[~amb], T[m].values[~amb]):.4f}")
# ---------- handcrafted features ----------
T["phase2"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
V["phase2"] = np.where(V.contrast == False, "non_contrast", V.phase.fillna("unknown"))
HUF = ["fluid_ml", "fluid_frac", "fluid_maxcc_ml", "fluid_cc10_ml", "fluid_ncc10", "fluid_cc10_hu", "pleural_fluid_ml", "hd_fluid_ml",
       "cav_ml_-20_0", "cav_ml_0_15", "cav_ml_15_30", "cav_ml_30_45", "cav_ml_45_70", "cav_vox",
       "periliver_ml", "periliver_frac", "periliver_hu_med", "perispleen_ml", "perispleen_frac", "perispleen_hu_med",
       "pelvic_fluid_ml", "upper_fluid_ml", "peribladder_ml", "peribladder_frac", "subq_fat_hu", "subq_edema_frac", "subq_fat_ml",
       "mes_fat_hu", "mes_fat_ml", "bowel_ml", "bowel_fluid_ml", "bowel_dense_ml", "bowel_gas_ml",
       "liver_vox", "spleen_vox", "bladder_vox", "liver_hu", "spleen_hu", "aorta_hu", "portal_hu", "kidney_r_hu", "pelvis_cov", "abd_z_mm", "liver_cut_top"]
HUF = [c for c in HUF if c in T]
for c in HUF:
    a = auc(y, T[c]); rec("2_feature_auc", c, y.sum(), (1 - y).sum(), max(a, 1 - a) if a == a else a,
                         f"dir={'+' if a >= .5 else '-'}; med pos={T[c][y==1].median():.3g} neg={T[c][y==0].median():.3g}")
LOGC = [c for c in HUF if c.endswith(("_ml", "_vox", "ncc10"))]
med = T[HUF].median()
def X(d, cols, extra=()):
    x = d[cols].copy().astype(float)
    for c in cols:
        if c in LOGC: x[c] = np.log1p(x[c].clip(lower=0))
    x = x.fillna(np.log1p(med[cols].clip(lower=0)).where([c in LOGC for c in cols], med[cols]))
    for e in extra: x[e] = d[e].values
    return x
H = T.fluid_ml.notna().values; TH = T[H].reset_index(drop=True); yH = y[H]
rec("3_stack_oof", "HU subset: E3_late AUROC (same rows as stacks)", yH.sum(), (1 - yH).sum(), auc(yH, TH.E3_late))
rec("3_stack_oof", "HU subset: E0 AUROC", yH.sum(), (1 - yH).sum(), auc(yH, TH.E0))
folds = TH.fold.values
def oof_fit(cols, extra=(), model="lr", C=0.3):
    T, y = TH, yH
    p = np.zeros(len(T))
    for k in range(5):
        a, b = folds != k, folds == k
        if model == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
        Xa = X(T[a], cols, extra); m.fit(Xa, y[a])
        p[b] = m.decision_function(X(T[b], cols, extra)) if model == "lr" else m.predict_proba(X(T[b], cols, extra))[:, 1]
    return p
T["female"] = (T.Gender == "Female").astype(int); V["female"] = (V.Gender == "Female").astype(int)
T["age"] = T.Age.fillna(T.Age.median()); V["age"] = V.Age.fillna(T.Age.median())
TH = T[H].reset_index(drop=True)
FLU = [c for c in HUF if c not in ("liver_vox", "spleen_vox", "liver_hu", "spleen_hu", "aorta_hu", "portal_hu", "kidney_r_hu")]
WL = [c for c in ["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"] if c in T]
P = {}
P["hu_lr"] = oof_fit(HUF); P["hu_gbm"] = oof_fit(HUF, model="gbm")
P["fluid_only_lr"] = oof_fit(FLU)
P["e3_hu_lr"] = oof_fit(HUF, ["E3_late"]); P["e3_hu_gbm"] = oof_fit(HUF, ["E3_late"], model="gbm")
P["e3_fluid_lr"] = oof_fit(FLU, ["E3_late"])
P["e3_hu_age_sex"] = oof_fit(HUF, ["E3_late", "age", "female"])
P["win_lr"] = oof_fit([], WL + ["E3_late"])
P["win_hu_lr"] = oof_fit(HUF, WL + ["E3_late"])
if "E3_late_all12" in T:
    P["all12_hu_lr"] = oof_fit(HUF, ["E3_late_all12"])
P["age_only"] = TH.age.values
P["age_sex"] = oof_fit([], ["age", "female"])
for k, p in P.items():
    T["p_" + k] = np.nan; T.loc[H, "p_" + k] = p; TH["p_" + k] = p
    rec("3_stack_oof", k, yH.sum(), (1 - yH).sum(), auc(yH, p))
best = "p_e3_hu_gbm" if auc(yH, TH.p_e3_hu_gbm) > auc(yH, TH.p_e3_hu_lr) else "p_e3_hu_lr"
rec("3_stack_oof", "chosen stack = " + best, yH.sum(), (1 - yH).sum(), auc(yH, TH[best]))
# bootstrap CI of the stacking gain (HU subset)
rng = np.random.default_rng(0); gs = []
for _ in range(300):
    i = rng.integers(0, len(yH), len(yH)); gs.append(auc(yH[i], TH[best].values[i]) - auc(yH[i], TH.E3_late.values[i]))
rec("3_stack_oof", "stack - E3_late gain (bootstrap 95% CI)", yH.sum(), (1 - yH).sum(), np.mean(gs), f"CI=[{np.quantile(gs,.025):.4f},{np.quantile(gs,.975):.4f}]")
# ---------- val descriptive: fit on all train, apply to val ----------
def full_fit(cols, extra=(), model="lr"):
    m = (make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")) if model == "lr"
         else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))
    m.fit(X(TH, cols, extra), yH); Xv = X(V, cols, extra)
    return m.decision_function(Xv) if model == "lr" else m.predict_proba(Xv)[:, 1]
V = V[V.fluid_ml.notna()].reset_index(drop=True)
yv = V.y_asc.values.astype(int); vd = []
for nm, s in ([] if len(V) < 50 else [("E0", V.E0), ("E3_late", V.E3_late), ("hu_lr", full_fit(HUF)), ("e3_hu_lr", full_fit(HUF, ["E3_late"])),
              ("e3_hu_gbm", full_fit(HUF, ["E3_late"], "gbm"))]) + ([("E3_late_all12", V.E3_late_all12)] if "E3_late_all12" in V else []):
    vd.append(dict(item=nm, n_pos=int(yv.sum()), n_neg=int((1 - yv).sum()), auroc_val_descriptive=round(auc(yv, s), 4)))
    print("VAL descriptive", nm, round(auc(yv, s), 4))
pd.DataFrame(vd).to_csv(A + "val_descriptive.csv", index=False)
# ---------- 4. amount subgroups (pos subgroup vs all negatives) ----------
neg = y == 0
def pos_sub(sec, name, pm, negmask=neg, scores=("E3_late", best)):
    pm = np.asarray(pm) & (y == 1); mm = pm | negmask
    for sc in scores:
        rec(sec, f"{name} [{sc}]", pm.sum(), negmask.sum(), auc(y[mm], T[sc].values[mm]), f"frac_of_pos={pm.sum()/y.sum():.3f}")
T["E3_late_onHU"] = np.where(H, T.E3_late, np.nan)
SC4 = tuple(c for c in ["E3_late", "E3_late_all12", "E3_late_onHU", best, "p_all12_hu_lr"] if c in T)
for a in ["trace", "small", "moderate", "large", "not_stated"]:
    pos_sub("4_amount", "amount=" + a, T.amount == a, scores=SC4)
pos_sub("4_amount", "category probably_yes", T.asc_cat == "probably_yes")
pos_sub("4_amount", "hemorrhagic=yes", T.hemorrhagic == "yes")
for k in ["kw_trace", "kw_pelvic", "kw_perihep", "kwp_loculated", "kw_physiologic", "kw_uncertain", "kw_postop", "kwp_hemo", "kw_free_fluid", "kw_ascites_word"]:
    pos_sub("4_pos_kw", k + "=1", T[k] == 1, scores=("E3_late",))
pelvic_only = (T.kw_pelvic == 1) & (T.kw_perihep == 0) & (T.amount.isin(["trace", "small"]))
pos_sub("4_pos_kw", "pelvic-only trace/small", pelvic_only, scores=SC4)
pos_sub("4_pos_kw", "pelvis not covered by scan (no bladder/hips)", T.pelvis_cov == 0)
pos_sub("4_pos_kw", "spleen mask <100 vox", T.spleen_vox < 100)
# amount mix + loss decomposition: AUROC if each amount group were scored like moderate/large
for a in ["trace", "small", "moderate", "large", "not_stated"]:
    mk = (y == 1) & (T.amount == a)
    rec("4_amount_mix", f"frac of pos amount={a}", mk.sum(), 0, mk.mean() / y.mean())
    rec("4_amount_mix", f"fluid_cc10_ml median amount={a}", mk.sum(), 0, T.fluid_cc10_ml[mk].median())
rec("4_amount_mix", "fluid_cc10_ml median neg", 0, neg.sum(), T.fluid_cc10_ml[neg].median())
# ---------- 5. negatives composition / label noise ----------
nsub = {"neg cat=no": T.asc_cat == "no", "neg cat=probably_no": T.asc_cat == "probably_no", "neg cat=not_mentioned": T.asc_cat == "not_mentioned",
        "neg amount=trace (label says trace but neg)": T.amount == "trace", "neg amount=small": T.amount == "small",
        "neg: non-negated fluid/ascites sentence": T.kwp_fluid == 1,
        "neg physiologic": T.kw_physiologic == 1, "neg pelvic fluid mention": (T.kw_pelvic == 1) & (T.kw_negated == 0),
        "neg loculated/collection/abscess": T.kwp_loculated == 1, "neg hemo/hematoma": T.kwp_hemo == 1, "neg postop": T.r_postop == 1,
        "neg pleural effusion": T.r_pleural == 1, "neg anasarca/edema": T.r_anasarca == 1, "neg cirrhosis": T.r_cirrhosis == 1,
        "neg splenomegaly label": T.y_splenomegaly == 1, "neg carcinomatosis": T.r_carcinomatosis == 1,
        "neg mesenteric edema": T.r_mesent_edema == 1, "neg bowel obstruction": T.y_bowel_obstruction == 1,
        "neg pancreatitis": T.r_pancreatitis == 1, "neg no oral contrast": T.r_no_oral == 1, "neg pregnancy": T.r_pregnan == 1,
        "neg ovarian/adnexal cyst": T.r_ovarian_cyst == 1, "neg heart failure signs": T.r_heart_fail == 1, "neg transplant": T.r_transplant == 1}
thr = {}
for sc in ["E3_late", best]:
    thr[sc] = np.nanquantile(T[sc][neg], 0.90)
for nm, mk in nsub.items():
    mk = np.asarray(mk) & neg
    mm = (y == 1) | mk
    rec("5_neg_sub", nm, y.sum(), mk.sum(), auc(y[mm], S[mm]),
        f"frac_of_neg={mk.sum()/neg.sum():.3f}; FPR@90spec E3={(S[mk]>thr['E3_late']).mean():.3f} stack={(T[best].values[mk & H]>thr[best]).mean():.3f}")
# label boundary: small positives vs small/trace negatives (same words, opposite labels)
for nm, nm2 in [("small", ["small"]), ("small", ["trace"]), ("small", ["trace", "small"])]:
    mm = ((y == 1) & (T.amount == nm).values) | (neg & T.amount.isin(nm2).values)
    for sc in ["E3_late", "fluid_cc10_ml", best]:
        rec("5_boundary", f"pos amount={nm} vs neg amount in {nm2} [{sc}]", ((y == 1) & mm).sum(), (neg & mm).sum(), auc(y[mm], T[sc].values[mm]))
mm = (neg & T.amount.isin(["trace", "small"]).values) | (neg & (T.asc_cat == "no").values)
yy = T.amount.isin(["trace", "small"]).values[mm].astype(int)
for sc in ["E3_late", "fluid_cc10_ml", best]:
    rec("5_boundary", f"within negatives: trace/small-labelled-neg vs cat=no [{sc}]", yy.sum(), (1 - yy).sum(), auc(yy, T[sc].values[mm]))
# systemic co-findings: positives WITHOUT pleural effusion / anasarca / cirrhosis / carcinomatosis
nocue = (T.r_pleural == 0) & (T.r_anasarca == 0) & (T.r_cirrhosis == 0) & (T.r_carcinomatosis == 0)
for sc in ["E3_late", best]:
    m1 = ((y == 1) & nocue.values) | neg
    rec("5_pos_conf", f"pos w/o any systemic cue (pleural/anasarca/cirrhosis/carcinomatosis) [{sc}]", ((y == 1) & nocue.values).sum(), neg.sum(), auc(y[m1], T[sc].values[m1]), f"frac_of_pos={(nocue[y==1]).mean():.3f}")
    m2 = nocue.values
    rec("5_pos_conf", f"both pos and neg w/o systemic cue [{sc}]", (y[m2] == 1).sum(), (y[m2] == 0).sum(), auc(y[m2], T[sc].values[m2]))
# exclusion deltas: AUROC(all minus negative subgroup) - AUROC(all)  (~ AUROC cost of that subgroup)
base = auc(y, S)
for nm in ["neg amount=trace (label says trace but neg)", "neg amount=small", "neg loculated/collection/abscess", "neg hemo/hematoma",
           "neg pleural effusion", "neg anasarca/edema", "neg cirrhosis", "neg bowel obstruction", "neg carcinomatosis", "neg postop", "neg splenomegaly label"]:
    mk = np.asarray(nsub[nm]) & neg
    rec("5_excl_delta", nm, y.sum(), (neg & ~mk).sum(), auc(y[~mk], S[~mk]) - base, f"frac_of_neg={mk.sum()/neg.sum():.3f}")
amb2 = neg & T.amount.isin(["trace", "small"]).values
rec("5_excl_delta", "neg trace+small (ambiguous label boundary)", y.sum(), (neg & ~amb2).sum(), auc(y[~amb2], S[~amb2]) - base, f"frac_of_neg={amb2.sum()/neg.sum():.3f}")
# positive subgroup counterfactual: small positives scored as separable as moderate ones (upper bound)
fs = ((y == 1) & (T.amount == "small").values).sum() / y.sum()
mm_s = ((y == 1) & (T.amount == "small").values) | neg; mm_m = ((y == 1) & (T.amount == "moderate").values) | neg
rec("5_excl_delta", "small positives at moderate-level AUROC (upper bound gain)", y.sum(), neg.sum(), fs * (auc(y[mm_m], S[mm_m]) - auc(y[mm_s], S[mm_s])), f"frac_of_pos={fs:.3f}")
# pos keyword AUROC (vs all negatives) for confounders
for nm, col in [("pos cirrhosis", "r_cirrhosis"), ("pos anasarca", "r_anasarca"), ("pos pleural", "r_pleural"), ("pos carcinomatosis", "r_carcinomatosis"),
                ("pos postop", "r_postop"), ("pos splenomegaly label", "y_splenomegaly")]:
    pos_sub("5_pos_conf", nm + "=1", T[col] == 1, scores=("E3_late",)); pos_sub("5_pos_conf", nm + "=0", T[col] == 0, scores=("E3_late",))
# ---------- 6. confounders ----------
for nm, col in [("cirrhosis kw", "r_cirrhosis"), ("splenomegaly label", "y_splenomegaly"), ("pleural effusion kw", "r_pleural"),
                ("anasarca kw", "r_anasarca"), ("carcinomatosis kw", "r_carcinomatosis"), ("female", "female")]:
    rec("6_conf_as_predictor", nm, y.sum(), neg.sum(), auc(y, T[col].fillna(0)),
        f"rate pos={T[col][y==1].mean():.3f} neg={T[col][neg].mean():.3f}")
rec("6_conf_as_predictor", "age only", y.sum(), neg.sum(), auc(y, T.age))
T["agebin"] = pd.cut(T.Age, [0, 30, 40, 50, 60, 70, 80, 120])
def strat_auc(sc, by):
    num = den = 0
    for _, g in T.groupby(by, observed=True):
        yy = g.y_asc.values.astype(int)
        if 0 < yy.sum() < len(yy):
            w = yy.sum() * (1 - yy).sum(); num += w * auc(yy, g[sc]); den += w
    return num / den
for sc in ["E0", "E3_late", best]:
    rec("6_age_matched", f"age-stratified AUROC (10y bins) {sc}", y.sum(), neg.sum(), strat_auc(sc, "agebin"))
    rec("6_age_matched", f"age x sex stratified AUROC {sc}", y.sum(), neg.sum(), strat_auc(sc, ["agebin", "Gender"]))
for col in ["agebin", "Gender", "phase2"]:
    for v in T[col].dropna().unique():
        mm = (T[col] == v).values
        rec("6_subgroup", f"{col}={v}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]), f"stack={auc(y[mm], T[best].values[mm]):.3f}; prev={y[mm].mean():.3f}")
# cirrhosis / splenomegaly shortcut: AUROC within strata
for col in ["r_cirrhosis", "y_splenomegaly", "r_pleural", "r_anasarca"]:
    for v in [0, 1]:
        mm = (T[col].fillna(0) == v).values
        rec("6_within", f"{col}={v}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]), f"stack={auc(y[mm], T[best].values[mm]):.3f}")
# young women, trace/pelvic
yw = (T.female == 1) & (T.Age < 45)
rec("6_within", "women <45: AUROC E3", y[yw].sum(), (1 - y[yw]).sum(), auc(y[yw], S[yw]), f"stack={auc(y[yw], T[best].values[yw]):.3f}")
rec("6_within", "women <45: trace/small frac of pos", y[yw].sum(), 0, T.amount[yw & (y == 1)].isin(["trace", "small"]).mean())
rec("6_within", "others: trace/small frac of pos", y[~yw].sum(), 0, T.amount[~yw & (y == 1)].isin(["trace", "small"]).mean())
# ---------- 7. errors at 90% specificity ----------
t = thr["E3_late"]; fn = (y == 1) & (S <= t); fp = neg & (S > t)
rec("7_err90", "sensitivity@90spec E3_late", y.sum(), neg.sum(), (S[y == 1] > t).mean())
rec("7_err90", "sensitivity@90spec stack (HU subset)", yH.sum(), (1 - yH).sum(), (TH[best].values[yH == 1] > thr[best]).mean())
rec("7_err90", "sensitivity@90spec E3_late (HU subset)", yH.sum(), (1 - yH).sum(), (TH.E3_late.values[yH == 1] > thr["E3_late"]).mean())
rec("7_err90", "sensitivity@90spec E0", y.sum(), neg.sum(), (T.E0[y == 1] > np.quantile(T.E0[neg], .9)).mean())
lowf = T.fluid_cc10_ml < T.fluid_cc10_ml[neg].median()
fn_cat = np.select([T.asc_cat == "probably_yes", T.kw_uncertain == 1, T.amount == "trace", T.amount == "small", pelvic_only, T.amount.isin(["moderate", "large"])],
                   ["A 不確定（probably_yes）", "B 報告用語不確定", "C 微量（trace）", "D 少量（small）", "E 只在骨盆", "F 中～大量仍漏掉"], "G 其他（未註明量）")
fp_cat = np.select([T.amount.isin(["trace", "small"]) | (T.kwp_fluid == 1), T.kwp_loculated == 1, T.kwp_hemo == 1,
                    T.r_pleural == 1, T.r_anasarca == 1, T.r_cirrhosis == 1, T.y_splenomegaly == 1, T.r_carcinomatosis == 1, T.r_postop == 1, T.y_bowel_obstruction == 1],
                   ["a 報告提到少量積液但標為陰性", "b 局限性積液／膿瘍", "c 血腫／出血", "d 肋膜積液", "e 全身水腫", "f 肝硬化", "g 脾腫大",
                    "h 腹膜轉移", "i 術後", "j 腸阻塞"], "k 其他")
T["fn_cat"] = np.where(fn, fn_cat, ""); T["fp_cat"] = np.where(fp, fp_cat, "")
for c in sorted(set(fn_cat)):
    rec("7_FN", c, fn.sum(), 0, (fn_cat[fn] == c).mean(), f"base rate in pos={(fn_cat[y==1]==c).mean():.3f}")
for c in sorted(set(fp_cat)):
    rec("7_FP", c, 0, fp.sum(), (fp_cat[fp] == c).mean(), f"base rate in neg={(fp_cat[neg]==c).mean():.3f}")
# FN / FP feature medians
for c in ["fluid_cc10_ml", "periliver_ml", "pelvic_fluid_ml", "pleural_fluid_ml", "subq_edema_frac", "spleen_vox"]:
    rec("7_feat", f"{c} median: FN / TP / FP / TN", fn.sum(), fp.sum(), T[c][fn].median(),
        f"TP={T[c][(y==1)&~fn].median():.3g} FP={T[c][fp].median():.3g} TN={T[c][neg&~fp].median():.3g}")
# ---------- 8. ceiling / cleaned-label ----------
clean = ~(((y == 1) & ((T.asc_cat == "probably_yes") | (T.kw_uncertain == 1))) | (neg & ((T.asc_cat == "probably_no") | T.amount.isin(["trace", "small"]) | (T.kwp_fluid == 1))))
for sc in ["E3_late", best]:
    rec("8_ceiling", f"AUROC on clean subset {sc}", y[clean].sum(), (1 - y[clean]).sum(), auc(y[clean], T[sc].values[clean]))
    nt = ~((y == 1) & (T.amount == "trace").values)
    rec("8_ceiling", f"AUROC excluding trace positives {sc}", y[nt].sum(), (1 - y[nt]).sum(), auc(y[nt], T[sc].values[nt]))
    nts = ~((y == 1) & T.amount.isin(["trace", "small"]).values)
    rec("8_ceiling", f"AUROC excluding trace+small positives {sc}", y[nts].sum(), (1 - y[nts]).sum(), auc(y[nts], T[sc].values[nts]))
pd.DataFrame(out).to_csv(A + "results_train_oof.csv", index=False)
keep = ["study_id", "fold", "y_asc", "asc_cat", "amount", "hemorrhagic"] + MODES + [c for c in T if c.startswith("p_")] + HUF + ["fn_cat", "fp_cat"]
T[keep].to_csv(A + "per_case_train_oof.csv", index=False)
print("ANALYZE_DONE best=", best)
