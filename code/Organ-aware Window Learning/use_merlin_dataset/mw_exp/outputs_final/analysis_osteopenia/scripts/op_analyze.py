"""Osteopenia error analysis on TRAIN 5-fold OOF (val descriptive only, no tuning; test never read)."""
import numpy as np, pandas as pd, json, warnings, glob
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import os
A = os.environ.get("OP_DIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_osteopenia/")
lab = pd.read_csv(A + "labels_trval.csv")
TRV = set(lab.study_id)
def rd(f):
    for g in (f, f + ".partial"):
        if os.path.exists(g) and os.path.getsize(g) > 10: return pd.read_csv(g)
    return pd.DataFrame({"study_id": []})
hu = pd.concat([rd(A + f) for f in ["hu_a6.csv", "hu_ct.csv", "hu_fa.csv"]]).drop_duplicates("study_id")
hu = hu[hu.study_id.isin(TRV)]
bun = pd.read_csv(A + "bundle_trval.csv").rename(columns={"aorta_hu_med": "aorta_hu"})   # scripts/op_bundle.py (train/val only)
oof = pd.read_csv(A + "oof_train.csv")
hu = hu.drop(columns=[c for c in hu.columns if c == "aorta_hu"])
df = lab.merge(bun, on="study_id", how="left").merge(hu, on="study_id", how="left")
# primary bone HU (full coverage): fracture_features.py L1 vertebral-body median HU (anterior 40%), fallback mean T12-L2
df["L1c"] = df["L1_hu"].fillna(df["T12_L2_hu"]); df["trab"] = df["T12_L2_hu"]
for c in ["L1_core_med", "L1_body_med", "L1_whole_med", "L1_core_p25", "autochthon_hu", "psoas_hu"]:
    if c not in df: df[c] = np.nan
ncols = [f"{l}_body_n1000" for l in ["T12", "L1", "L2", "L3", "L4", "L5"] if f"{l}_body_n1000" in df]
df["cement"] = (df[ncols].fillna(0).max(1) >= 15).astype(int) if ncols else 0
df["cement_known"] = df["L1_body_med"].notna() | df.get("L2_body_med", pd.Series(np.nan, index=df.index)).notna()
df["phase2"] = np.where(df.contrast == False, "non_contrast", df.phase.fillna("unknown"))
df["aorta_bin"] = pd.cut(df.aorta_hu, [-1e9, 70, 150, 250, 1e9], labels=["<70", "70-150", "150-250", ">=250"]).astype(str)
df["female"] = (df.Gender == "Female").astype(float)
df["agebin"] = pd.cut(df.Age, [0, 50, 60, 70, 80, 200], labels=["<50", "50-60", "60-70", "70-80", ">=80"]).astype(str)
T = df[df.split == "train"].merge(oof, on="study_id")
T = T[T.y_op.notna()].reset_index(drop=True)
y = T.y_op.values.astype(int); S = T.E3_late.values
print("train n", len(T), "pos", y.sum(), "HU coverage", T.L1c.notna().mean().round(4), flush=True)
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 and m.sum() > 10 else np.nan
out = []
def rec(sec, name, n1, n0, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
def recm(sec, name, mask, score, note=""):
    mask = np.asarray(mask, bool); rec(sec, name, y[mask].sum(), (1 - y[mask]).sum(), auc(y[mask], np.asarray(score)[mask]), note)
ALL = np.ones(len(T), bool)
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "L1unit_E0", "L1unit_E3_late"] if c in T]
for m in MODES:
    recm("1_overall_oof", m, ALL, T[m])
rec("1_overall_oof", "lumbar_spine emb missing frac (pos/neg)", 0, 0, np.nan, f"{T.L1c.isna()[y==1].mean():.3f}/{T.L1c.isna()[y==0].mean():.3f} (HU missing)")
# 2. single HU features (low HU -> osteopenia, so negate)
for f in ["L1c", "trab", "T12_hu_med", "L2_hu_med", "L3_hu_med", "L4_hu_med", "L5_hu_med", "min_body_hu_p10", "aorta_hu", "Age"]:
    sgn = 1 if f == "Age" else -1
    rec("2_single_feature", f, y.sum(), (1 - y).sum(), auc(y, sgn * T[f]), f"median pos={T[f][y==1].median():.1f} neg={T[f][y==0].median():.1f}")
rec("2_single_feature", "female", y.sum(), (1 - y).sum(), auc(y, T.female))
rec("2_single_feature", "fracture label", y.sum(), (1 - y).sum(), auc(y, T.y_frac.fillna(0)))
rec("2_single_feature", "min_height_nb_ratio (neg)", y.sum(), (1 - y).sum(), auc(y, -T.min_height_nb_ratio))
# core-vs-body-vs-whole on the subset where the voxel-level features are done (pseudo-random subset by id order)
sub = T.L1_core_med.notna().values
rec("2b_core_subset", "subset n (core HU available)", y[sub].sum(), (1 - y[sub]).sum(), sub.mean())
for f in ["L1_core_med", "L1_core_mean", "L1_core_p25", "L1_body_med", "L1_body_p10", "L1_whole_med", "L1_whole_mean", "L1c", "autochthon_hu", "psoas_hu"]:
    if f in T and sub.sum() > 100:
        recm("2b_core_subset", f"-{f}", sub, -T[f].values, f"median pos={T[f][sub & (y==1)].median():.1f} neg={T[f][sub & (y==0)].median():.1f}")
if sub.sum() > 100:
    recm("2b_core_subset", "E3_late (same subset)", sub, S)
    T["_core_minus_muscle"] = T.L1_core_med - T.autochthon_hu
    recm("2b_core_subset", "-(L1 core - paraspinal muscle)", sub, -T._core_minus_muscle.values)
    rec("2b_core_subset", "corr(L1 core, L1c body)", 0, 0, T[["L1_core_med", "L1c"]].dropna().corr().iloc[0, 1])
rec("2_single_feature", "corr(E3_late, -L1c)", 0, 0, np.corrcoef(S[T.L1c.notna()], -T.L1c[T.L1c.notna()])[0, 1])
rec("2_single_feature", "corr(E3_late, Age)", 0, 0, np.corrcoef(S[T.Age.notna()], T.Age[T.Age.notna()])[0, 1])
# 3. phase: HU offset by phase in negatives; AUROC by phase; phase-corrected HU (fold-wise OLS on train-fold negatives)
for ph in T.phase2.dropna().unique():
    mm = (T.phase2 == ph).values
    if y[mm].sum() >= 15:
        recm("3_phase", f"E3_late | phase={ph}", mm, S, f"L1c AUROC={auc(y[mm], -T.L1c[mm]):.3f}; L1c median neg={T.L1c[mm & (y==0)].median():.0f} pos={T.L1c[mm & (y==1)].median():.0f}; aorta med={T.aorta_hu[mm].median():.0f}")
    else:
        rec("3_phase", f"E3_late | phase={ph} (n_pos<15)", y[mm].sum(), (1 - y[mm]).sum(), np.nan, f"n={mm.sum()}")
for b in ["<70", "70-150", "150-250", ">=250"]:
    mm = (T.aorta_bin == b).values
    recm("3_phase", f"E3_late | aorta HU {b}", mm, S, f"L1c AUROC={auc(y[mm], -T.L1c[mm]):.3f}; L1c med neg={T.L1c[mm & (y==0)].median():.0f}; frac cases={mm.mean():.3f}")
def phase_corr(col):
    v = np.full(len(T), np.nan)
    X = pd.DataFrame({"a": T.aorta_hu.fillna(T.aorta_hu.median()).clip(0, 500), "nc": (T.phase2 == "non_contrast").astype(float)})
    for k in range(5):
        tr = ((T.fold != k) & (y == 0) & T[col].notna()).values; te = (T.fold == k).values
        lr = LinearRegression().fit(X[tr], T[col][tr]); v[te] = T[col].values[te] - lr.predict(X[te]) + T[col][tr].median()
    return v
T["L1c_pc"] = phase_corr("L1c")
rec("3_phase", "L1c phase-corrected (aorta HU + non-contrast, fold-wise)", y.sum(), (1 - y).sum(), auc(y, -T.L1c_pc))
lr0 = LinearRegression().fit(pd.DataFrame({"a": T.aorta_hu.fillna(150).clip(0, 500), "nc": (T.phase2 == "non_contrast").astype(float)})[(y == 0) & T.L1c.notna().values],
                             T.L1c[(y == 0) & T.L1c.notna().values])
rec("3_phase", "OLS slope L1c per 100 HU aorta (neg)", 0, 0, lr0.coef_[0] * 100, f"non-contrast offset={lr0.coef_[1]:.1f} HU")
# 4. label definition / under-reporting
for thr in [80, 100, 110, 130, 160]:
    rec("4_label", f"frac L1c<{thr}: neg", 0, (y == 0).sum(), (T.L1c[y == 0] < thr).mean(), f"pos={(T.L1c[y==1] < thr).mean():.3f}; n_neg<thr={(T.L1c[y==0] < thr).sum()}; n_pos<thr={(T.L1c[y==1] < thr).sum()}")
for ab in ["<50", "50-60", "60-70", "70-80", ">=80"]:
    mm = (T.agebin == ab).values
    rec("4_label", f"age {ab}: report-positive rate", y[mm].sum(), (1 - y[mm]).sum(), y[mm].mean(),
        f"frac L1c<100 among neg={(T.L1c[mm & (y==0)] < 100).mean():.3f}; <130={(T.L1c[mm & (y==0)] < 130).mean():.3f}; E3 AUROC={auc(y[mm], S[mm]):.3f}")
lo = (T.L1c < 100).values; nm_ = (T.L1c >= 160).values
rec("4_label", "report-positive rate among L1c<100", y[lo].sum(), (1 - y[lo]).sum(), y[lo].mean())
rec("4_label", "report-positive rate among L1c 100-160", y[~lo & ~nm_ & T.L1c.notna().values].sum(), 0, y[~lo & ~nm_ & T.L1c.notna().values].mean())
rec("4_label", "report-positive rate among L1c>=160", y[nm_].sum(), (1 - y[nm_]).sum(), y[nm_].mean())
# AUROC vs negative subsets
for nmk, nmask in [("neg L1c>=160 (clearly normal)", (y == 0) & nm_), ("neg L1c 110-160", (y == 0) & (T.L1c >= 110).values & ~nm_),
                   ("neg L1c<110 (likely missed)", (y == 0) & (T.L1c < 110).values), ("neg L1c<100", (y == 0) & lo)]:
    mm = (y == 1) | nmask
    recm("4_label_negsubset", f"E3_late: pos vs {nmk}", mm, S, f"L1c AUROC={auc(y[mm], -T.L1c[mm]):.3f}")
mm = ~((y == 0) & (T.L1c < 110).values)
recm("4_label_negsubset", "E3_late after dropping neg with L1c<110", mm, S)
mm = ~((y == 0) & lo)
recm("4_label_negsubset", "E3_late after dropping neg with L1c<100", mm, S)
# positives with normal HU
for nmk, pm in [("pos L1c>=160", nm_), ("pos L1c 110-160", (T.L1c >= 110).values & ~nm_), ("pos L1c<110", (T.L1c < 110).values), ("pos L1c missing", T.L1c.isna().values)]:
    mm = (pm & (y == 1)) | (y == 0)
    recm("4_pos_by_HU", f"E3_late: {nmk} vs all neg", mm, S, f"frac of pos={(pm & (y==1)).sum()/y.sum():.3f}")
# 5. age / sex shortcut
rec("5_age_sex", "Age-only AUROC", y.sum(), (1 - y).sum(), auc(y, T.Age))
rec("5_age_sex", "female-only AUROC", y.sum(), (1 - y).sum(), auc(y, T.female))
rec("5_age_sex", "pos female frac / neg female frac", 0, 0, T.female[y == 1].mean(), f"neg={T.female[y==0].mean():.3f}")
rec("5_age_sex", "Age median pos / neg", 0, 0, T.Age[y == 1].median(), f"neg={T.Age[y==0].median():.1f}")
def strat_auc(score, keys):
    num = den = 0
    for _, g in T.assign(_s=score, _y=y).groupby(keys):
        p, n = (g._y == 1).sum(), (g._y == 0).sum()
        if p and n and len(g) > 10:
            num += auc(g._y, g._s) * p * n; den += p * n
    return num / den
T["age5"] = (T.Age // 5).clip(upper=19)
for nmk, sc in [("E3_late", S), ("E0", T.E0.values), ("-L1c", -T.L1c.fillna(T.L1c.median()).values)]:
    rec("5_age_sex", f"age(5y)xsex-matched AUROC {nmk}", y.sum(), (1 - y).sum(), strat_auc(sc, ["age5", "Gender"]))
    rec("5_age_sex", f"age(5y)-matched AUROC {nmk}", y.sum(), (1 - y).sum(), strat_auc(sc, ["age5"]))
for ab in ["<50", "50-60", "60-70", "70-80", ">=80"]:
    for sx in ["Female", "Male"]:
        mm = ((T.agebin == ab) & (T.Gender == sx)).values
        if y[mm].sum() >= 10:
            recm("5_age_sex_strata", f"E3_late | {sx} {ab}", mm, S, f"L1c AUROC={auc(y[mm], -T.L1c[mm]):.3f}; pos rate={y[mm].mean():.3f}")
# 6. co-reporting
for k in ["kw_compression", "kw_kypho", "kw_degen", "kw_scoliosis", "kw_hardware", "kw_bone_mets", "kw_mild", "kw_moderate", "kw_severe", "kw_diffuse", "kw_osteoporosis", "kw_demineral"]:
    pm = (T[k] == 1).values
    rec("6_coreport", f"{k}: rate pos / neg", 0, 0, pm[y == 1].mean(), f"neg={pm[y==0].mean():.4f}; E3 AUROC pos[{k}] vs all neg={auc(y[(pm & (y==1)) | (y==0)], S[(pm & (y==1)) | (y==0)]):.3f}; pos[~{k}]={auc(y[(~pm & (y==1)) | (y==0)], S[(~pm & (y==1)) | (y==0)]):.3f}")
pm = (T.y_frac == 1).values
rec("6_coreport", "fracture label: rate pos / neg", 0, 0, pm[y == 1].mean(), f"neg={pm[y==0].mean():.4f}; AUROC pos w/ fracture={auc(y[(pm & (y==1)) | (y==0)], S[(pm & (y==1)) | (y==0)]):.3f}; w/o={auc(y[(~pm & (y==1)) | (y==0)], S[(~pm & (y==1)) | (y==0)]):.3f}")
ck = T.cement_known.values.astype(bool); pm = (T.cement == 1).values
rec("6_coreport", "cement (>=15 vox >1000HU in a body; voxel subset): rate pos / neg", 0, 0, pm[ck & (y == 1)].mean() if (ck & (y==1)).any() else np.nan, f"neg={pm[ck & (y==0)].mean() if (ck & (y==0)).any() else np.nan:.4f}")
pm = (T.y_frac == 1).values
mm = (y == 0); rec("6_coreport", "neg with fracture: E3 score pct-rank among neg", 0, (pm & mm).sum(), pd.Series(S[mm]).rank(pct=True)[T.y_frac.values[mm] == 1].mean())
# 7. stacking (same folds)
HU = ["L1c", "trab", "T12_hu_med", "L2_hu_med", "L3_hu_med", "L4_hu_med", "min_body_hu_p10", "aorta_hu", "L1c_pc"]
X = T[HU].copy(); X["hu_missing"] = T.L1c.isna().astype(float); X["noncon"] = (T.phase2 == "non_contrast").astype(float)
X = X.fillna(X.median())
DEMO = pd.DataFrame({"age": T.Age.fillna(T.Age.median()), "female": T.female})
def oof_lr(cols_df, C=0.3):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = (T.fold != k).values, (T.fold == k).values
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        mdl.fit(cols_df[tr], y[tr]); p[te] = mdl.decision_function(cols_df[te])
    return p
T["p_L1c"] = oof_lr(X[["L1c", "hu_missing"]])
T["p_hu"] = oof_lr(X)
T["p_demo"] = oof_lr(DEMO)
T["p_hu_demo"] = oof_lr(pd.concat([X, DEMO], axis=1))
T["p_e3_L1c"] = oof_lr(pd.concat([X[["L1c", "hu_missing"]], T[["E3_late"]]], axis=1))
T["p_stack_e3"] = oof_lr(pd.concat([X, T[["E3_late"]]], axis=1))
T["p_stack_e0"] = oof_lr(pd.concat([X, T[["E0"]]], axis=1))
T["p_e3_demo"] = oof_lr(pd.concat([T[["E3_late"]], DEMO], axis=1))
T["p_stack_e3_demo"] = oof_lr(pd.concat([X, T[["E3_late"]], DEMO], axis=1))
allw = T[["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]]
T["p_stack_windows"] = oof_lr(allw)
T["p_stack_windows_hu"] = oof_lr(pd.concat([allw, X], axis=1))
for c in ["p_L1c", "p_hu", "p_demo", "p_hu_demo", "p_e3_L1c", "p_stack_e0", "p_stack_e3", "p_e3_demo", "p_stack_e3_demo", "p_stack_windows", "p_stack_windows_hu"]:
    rec("7_stack_oof", c, y.sum(), (1 - y).sum(), auc(y, T[c]), f"age(5y)xsex-matched={strat_auc(T[c].values, ['age5','Gender']):.3f}")
# window correlation
for w in ["E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]:
    rec("8_windows", f"corr(E0, {w}) OOF", 0, 0, np.corrcoef(T.E0, T[w])[0, 1], f"AUROC {w}={auc(y, T[w]):.3f}; corr with -L1c={np.corrcoef(T[w][T.L1c.notna()], -T.L1c.dropna())[0,1]:.3f}")
rec("8_windows", "corr(E0, -L1c)", 0, 0, np.corrcoef(T.E0[T.L1c.notna()], -T.L1c.dropna())[0, 1])
# 9. errors at 90% specificity (E3_late)
thr = np.quantile(S[y == 0], 0.90); pred = S >= thr
fn = (y == 1) & ~pred; fp = (y == 0) & pred; tp = (y == 1) & pred
rec("9_err", "sensitivity@90%spec E3_late", y.sum(), (1 - y).sum(), tp.sum() / y.sum())
for c in ["p_hu", "p_stack_e3", "p_stack_e3_demo"]:
    rec("9_err", f"sensitivity@90%spec {c}", y.sum(), (1 - y).sum(), (T[c][y == 1] >= np.quantile(T[c][y == 0], .9)).mean())
def cat_fn(r):
    if pd.isna(r.L1c): return "A 腰椎遮罩/HU 缺失"
    if r.kw_kypho: return "B 骨水泥（椎體成形術後）"
    if r.L1c >= 160: return "C L1 骨小樑 HU 正常（>=160）"
    if r.Age < 60: return "D 年輕（<60 歲）且 HU 偏低"
    if r.L1c >= 110: return "E HU 介於 110-160（輕度）"
    return "F HU<110 明確偏低，模型仍漏"
def cat_fp(r):
    if pd.isna(r.L1c): return "a 腰椎遮罩/HU 缺失"
    if r.kw_kypho: return "b 骨水泥（椎體成形術後）"
    if r.L1c < 110: return "c L1 HU<110：疑似漏標的骨質減少"
    if r.y_frac == 1 or r.kw_compression: return "d 合併骨折/壓迫性變形（未提骨質減少）"
    if r.L1c < 160: return "e HU 110-160（灰色地帶）"
    if r.Age >= 70: return "f HU 正常但高齡"
    return "g 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
bp = T[y == 1].apply(cat_fn, axis=1); bn = T[y == 0].apply(cat_fp, axis=1)
for k, v in T.fn_cat[fn].value_counts().items():
    rec("9_err_FN", k, int(v), 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(bp==k).mean():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    rec("9_err_FP", k, 0, int(v), v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(bn==k).mean():.3f}")
rec("9_err", "FN median Age / L1c", fn.sum(), 0, T.Age[fn].median(), f"L1c={T.L1c[fn].median():.0f}; TP Age={T.Age[tp].median():.0f} L1c={T.L1c[tp].median():.0f}")
rec("9_err", "FP median Age / L1c", 0, fp.sum(), T.Age[fp].median(), f"L1c={T.L1c[fp].median():.0f}; TN Age={T.Age[(y==0)&~pred].median():.0f} L1c={T.L1c[(y==0)&~pred].median():.0f}")
# 10. ceiling: label-noise simulation. Treat negatives with L1c<100 as unknown -> remove; and best HU+E3 stack on clean set.
for thr_ in [100, 110]:
    mm = ~((y == 0) & (T.L1c < thr_).values)
    for c in ["E3_late", "p_stack_e3"]:
        recm("10_ceiling", f"{c} without neg L1c<{thr_}", mm, T[c])
# 11. core-HU (trabecular, voxel subset) based label-definition analysis + label-noise ceiling
sub = T.L1_core_med.notna().values
if sub.sum() > 500:
    c = T.L1_core_med.values
    for thr_ in [80, 100, 110, 130, 160]:
        rec("11_core_label", f"frac L1 core<{thr_}: neg", 0, (sub & (y == 0)).sum(), (c[sub & (y == 0)] < thr_).mean(),
            f"pos={(c[sub & (y==1)] < thr_).mean():.3f}; report-positive rate among core<{thr_}={y[sub & (c < thr_)].mean():.3f} (n={(sub & (c < thr_)).sum()})")
    for ab in ["<50", "50-60", "60-70", "70-80", ">=80"]:
        mm = sub & (T.agebin == ab).values
        rec("11_core_label", f"age {ab}: frac neg core<110 / <160", 0, (mm & (y == 0)).sum(), (c[mm & (y == 0)] < 110).mean(),
            f"<160={(c[mm & (y==0)] < 160).mean():.3f}; pos core<160={(c[mm & (y==1)] < 160).mean():.3f}; n_neg core<160={(c[mm & (y==0)] < 160).sum()} vs n_pos={y[mm].sum()}")
    for thr_ in [110, 130, 160]:
        pi = (c[sub & (y == 0)] < thr_).mean(); rho = (c[sub & (y == 1)] >= thr_).mean()
        ceil = (1 - rho) * (1 - pi) + 0.5 * ((1 - rho) * pi + rho * (1 - pi))
        rec("11_ceiling_noise", f"perfect core<{thr_} detector vs report labels", (sub & (y==1)).sum(), (sub & (y==0)).sum(), ceil,
            f"pi(neg below thr)={pi:.3f}; rho(pos above thr)={rho:.3f}; AUROC of -core itself={auc(y[sub], -c[sub]):.3f}")
    for thr_ in [110, 130, 160]:
        mm = sub & ~((y == 0) & (c < thr_))
        recm("11_ceiling_noise", f"E3_late on subset after dropping neg core<{thr_}", mm, S, f"vs subset all={auc(y[sub], S[sub]):.3f}; n dropped={(sub & (y==0) & (c < thr_)).sum()}")
    # stacks on the subset (fold-wise LR restricted to subset)
    Ts = T[sub].reset_index(drop=True); ys = y[sub]
    def oof_sub(cols):
        p = np.zeros(len(Ts)); Xs = Ts[cols].fillna(Ts[cols].median())
        for k in range(5):
            tr, te = (Ts.fold != k).values, (Ts.fold == k).values
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(Xs[tr], ys[tr])
            p[te] = mdl.decision_function(Xs[te])
        return p
    CORE = ["L1_core_med", "L1_core_p25", "L1_body_p10", "L1c", "min_body_hu_p10", "autochthon_hu", "psoas_hu", "aorta_hu"]
    for nm, cols in [("E3_late only", ["E3_late"]), ("core HU feats", CORE), ("core HU + muscle + age/sex", CORE + ["Age", "female"]),
                     ("E3_late + core HU + muscle", CORE + ["E3_late"])]:
        p = oof_sub(cols); Ts["_p"] = p
        num = den = 0
        for _, g in Ts.assign(_y=ys).groupby(["age5", "Gender"]):
            a_, b_ = (g._y == 1).sum(), (g._y == 0).sum()
            if a_ and b_ and len(g) > 10: num += auc(g._y, g._p) * a_ * b_; den += a_ * b_
        rec("11_core_stack", nm, ys.sum(), (1 - ys).sum(), auc(ys, p), f"age(5y)xsex-matched={num/den:.3f}")
# FP vs FN: does imaging (HU) side with the model or with the label?
rec("9_err", "P(L1c of FP < L1c of FN)", fn.sum(), fp.sum(), auc(np.r_[np.ones(fp.sum()), np.zeros(fn.sum())], -np.r_[T.L1c[fp], T.L1c[fn]]),
    f"FP fracture-label rate={T.y_frac[fp].mean():.3f}; FN fracture-label rate={T.y_frac[fn].mean():.3f}; FN kw_mild={T.kw_mild[fn].mean():.3f}")
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
T[["study_id", "fold", "y_op", "op_cat", "y_frac", "Age", "Gender", "phase2", "aorta_hu", "L1c", "L1c_pc", "L1_core_med", "cement",
   "E0", "E3_late", "E1_bone", "p_hu", "p_stack_e3", "fn_cat", "fp_cat"]].to_csv(A + "per_case_train_oof.csv", index=False)
# VAL descriptive (models fit on full train OOF features; no tuning)
V = df[(df.split == "val") & df.y_op.notna()].merge(pd.read_csv(A + "scores_val_descriptive.csv"), on="study_id")
yv = V.y_op.values.astype(int)
Xv = V[[c for c in HU if c != "L1c_pc"]].copy()
lrv = LinearRegression().fit(pd.DataFrame({"a": T.aorta_hu.fillna(T.aorta_hu.median()).clip(0, 500), "nc": (T.phase2 == "non_contrast").astype(float)})[(y == 0) & T.L1c.notna().values], T.L1c[(y == 0) & T.L1c.notna().values])
Xv["L1c_pc"] = V.L1c - lrv.predict(pd.DataFrame({"a": V.aorta_hu.fillna(T.aorta_hu.median()).clip(0, 500), "nc": (V.phase2 == "non_contrast").astype(float)})) + T.L1c[y == 0].median()
Xv["hu_missing"] = V.L1c.isna().astype(float); Xv["noncon"] = (V.phase2 == "non_contrast").astype(float)
Xv = Xv[X.columns].fillna(X.median())
vrows = []
for nm in [c for c in ["E0", "E3_late", "E1_bone"] if c in V]:
    vrows.append(dict(model=nm, auc=auc(yv, V[nm]), n_pos=yv.sum()))
vrows.append(dict(model="-L1c", auc=auc(yv, -V.L1c), n_pos=yv.sum()))
vrows.append(dict(model="-L1c phase-corrected", auc=auc(yv, -Xv.L1c_pc), n_pos=yv.sum()))
for nm, cols in [("HU-LR", X.columns.tolist()), ("E3_late+HU stack", X.columns.tolist() + ["E3_late"])]:
    Xt = pd.concat([X, T[["E3_late"]]], axis=1)[cols]; Xvv = pd.concat([Xv, V[["E3_late"]]], axis=1)[cols]
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(Xt, y)
    vrows.append(dict(model=nm, auc=auc(yv, mdl.decision_function(Xvv)), n_pos=yv.sum()))
vrows.append(dict(model="val frac neg L1c<100", auc=(V.L1c[yv == 0] < 100).mean(), n_pos=0))
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 400)
print(res.to_string()); print(pd.DataFrame(vrows).round(4).to_string())
