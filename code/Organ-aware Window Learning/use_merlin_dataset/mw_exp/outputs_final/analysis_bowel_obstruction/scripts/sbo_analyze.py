"""Bowel-obstruction error analysis on TRAIN OOF (val descriptive only, no tuning on val, test never read)."""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = os.environ.get("ADIR", R + "analysis_bowel_obstruction/")
lab = pd.read_csv(A + "labels_trval.csv")
ft = pd.concat([pd.read_csv(A + f) for f in ("feat_a6.csv", "feat_ct.csv") if os.path.exists(A + f)]).drop_duplicates("study_id")
ft = ft[ft.err == 0] if "err" in ft else ft
# meta_trval.csv / expB_val_scores.csv: train/val-only extracts made on A6000 (metadata.csv filtered to labels_trval ids;
# expB E0/E3_late val probabilities) so the analysis can run on the Mac
if os.path.exists(A + "meta_trval.csv"):
    meta = pd.read_csv(A + "meta_trval.csv")
else:
    meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase"]]
df = lab.merge(ft, on="study_id", how="left").merge(meta, on="study_id", how="left")
oof = pd.read_csv(A + "oof_train.csv")
T = df[df.split == "train"].merge(oof, on="study_id"); T = T[T.y_sbo.notna()].reset_index(drop=True)
U = df[(df.split == "train") & df.y_sbo.isna()].merge(oof, on="study_id")      # unclear (excluded from AUROC)
y = T.y_sbo.values.astype(int); S = T.E3_late.values
pos, neg = y == 1, y == 0
def auc(yy, s, w=None):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    if len(set(yy[m])) < 2: return np.nan
    return roc_auc_score(yy[m], s[m], sample_weight=None if w is None else np.asarray(w)[m])
out = []
def rec(sec, name, n1, n0, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "E0_st", "E3_late_st"] if c in T]
for m in MODES: rec("1_overall_oof", m, pos.sum(), neg.sum(), auc(y, T[m]))
pct = pd.Series(S[neg]).rank(pct=True).values      # percentile of each negative among negatives
negpct = np.full(len(T), np.nan); negpct[neg] = pct
def sub(sec, name, pm, s=S):
    pm = np.asarray(pm, bool); mm = (pm & pos) | neg
    rec(sec, name, (pm & pos).sum(), neg.sum(), auc(y[mm], s[mm]), f"frac_of_pos={(pm & pos).sum() / pos.sum():.3f}")
def negsub(sec, name, nm, s=S):
    nm = np.asarray(nm, bool) & neg; mm = pos | nm
    rec(sec, name, pos.sum(), nm.sum(), auc(y[mm], s[mm]),
        f"frac_of_neg={nm.sum() / neg.sum():.3f}; mean pct among neg={np.nanmean(negpct[nm]):.3f}; FP rate@90spec={(s[nm] >= np.quantile(s[neg], .9)).mean():.3f}")
# 0 prevalence of text flags
KW = [c for c in T.columns if c.startswith(("kw_", "rep_"))]
for k in KW: rec("0_prev", k, pos.sum(), neg.sum(), T[k][pos].mean(), f"neg {T[k][neg].mean():.3f}; unclear {U[k].mean():.3f}")
# 2 subgroups among positives (pos-subgroup vs ALL neg)
for v in ["SBO", "LBO", "both", "not_stated"]: sub("2_type", f"type={v}", (T.sbo_type == v).values)
sub("2_type", "gastric outlet / duodenal / afferent(biliopancreatic) limb", T.kw_gastric_outlet == 1)
sub("2_type", "LBO or both or colonic-obstruction keyword", ((T.sbo_type.isin(["LBO", "both"])) | (T.kw_lbo == 1)).values)
for v in ["yes", "probably_yes"]: sub("3_grade", f"category={v}", (T.sbo_cat == v).values)
HG = (T.kw_highgrade == 1).values; PA = (T.kw_partial == 1).values
sub("3_grade", "high-grade/complete (quote)", HG & ~PA)
sub("3_grade", "partial/low-grade/early (quote)", PA)
sub("3_grade", "grade not stated", ~HG & ~PA)
sub("3_grade", "closed-loop/volvulus/internal hernia", T.kw_closedloop == 1)
sub("3_grade", "transition point stated", (T.kw_transition == 1) & (T.kw_no_transition == 0))
sub("3_grade", "NO transition point stated", T.kw_no_transition == 1)
sub("3_grade", "ileus/pseudo-obstruction mentioned", T.kw_ileus == 1)
sub("3_grade", "resolving/improving", T.kw_resolving == 1)
sub("3_grade", "uncertain wording (quote)", T.kw_uncertain == 1)
q = T.quote_max_mm
sub("3_grade", "quoted max diameter <35 mm", (q < 35).values); sub("3_grade", "quoted 35-45 mm", ((q >= 35) & (q < 45)).values); sub("3_grade", "quoted >=45 mm", (q >= 45).values)
for k, nm in [("kw_adhesion", "adhesion"), ("kw_hernia", "hernia"), ("kw_mass", "mass/malignancy/carcinomatosis"), ("kw_feces", "small-bowel feces sign"),
              ("kw_airfluid", "air-fluid levels"), ("kw_fluidfilled", "fluid-filled loops"), ("kw_intussusception", "intussusception"),
              ("rep_postop", "post-op report"), ("rep_ostomy", "ostomy/stoma"), ("rep_ngtube", "NG/enteric tube"),
              ("rep_gastroenteritis", "enteritis/colitis"), ("rep_ascites", "ascites/free fluid"), ("rep_bowel_wall", "bowel wall thickening/ischemia")]:
    sub("4_etiology_context", f"{nm}=1", T[k] == 1); sub("4_etiology_context", f"{nm}=0", T[k] == 0)
# 5 negatives: mimics, label confusion, shortcuts (all pos vs neg-subgroup)
negsub("5_neg_mimic", "neg: report affirms 'obstruct' (possible missed pos)", T.rep_aff_obstruct == 1)
negsub("5_neg_mimic", "neg: report affirms ileus/pseudo-obstruction", T.rep_aff_ileus == 1)
negsub("5_neg_mimic", "neg: report affirms dilated small bowel", T.rep_aff_sb_dilated == 1)
negsub("5_neg_mimic", "neg: report affirms dilated colon", T.rep_aff_colon_dilated == 1)
negsub("5_neg_mimic", "neg: any of the 4 above", (T[["rep_aff_obstruct", "rep_aff_ileus", "rep_aff_sb_dilated", "rep_aff_colon_dilated"]].sum(1) > 0))
negsub("5_neg_mimic", "neg: none of the 4 above", (T[["rep_aff_obstruct", "rep_aff_ileus", "rep_aff_sb_dilated", "rep_aff_colon_dilated"]].sum(1) == 0))
for k, nm in [("rep_postop", "post-op"), ("rep_ostomy", "ostomy"), ("rep_ngtube", "NG tube"), ("rep_gastroenteritis", "enteritis/colitis"),
              ("kw_hernia", "hernia"), ("rep_ascites", "ascites"), ("rep_oral_contrast", "oral contrast mentioned"), ("rep_bowel_wall", "wall thickening"),
              ("y_fa", "free air label"), ("y_asc", "ascites label"), ("rep_neg_sbo", "report explicitly says no obstruction")]:
    negsub("5_neg_mimic", f"neg: {nm}", T[k] == 1)
for c in ["no", "probably_no", "not_mentioned"]: negsub("5_neg_mimic", f"neg category={c}", T.sbo_cat == c)
for k, nm in [("rep_postop", "post-op"), ("rep_ngtube", "NG tube"), ("kw_hernia", "hernia"), ("rep_ostomy", "ostomy")]:
    for m in ["E0", "E3_late"]:
        rec("6_shortcut", f"{m}: neg-only AUROC {nm} vs not (higher = model scores {nm})", 0, neg.sum(), auc(T[k][neg], T[m][neg]))
    rec("6_shortcut", f"{nm} flag as score (all)", pos.sum(), neg.sum(), auc(y, T[k]))
    mm = (T[k] == 1).values; rec("6_shortcut", f"E3 AUROC within {nm}=1", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
    mm = (T[k] == 0).values; rec("6_shortcut", f"E3 AUROC within {nm}=0", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
if "esoph_hi_vox" in T:
    rec("6_shortcut", "esoph_hi_vox (enteric-tube proxy) as score", pos.sum(), neg.sum(), auc(y, T.esoph_hi_vox))
    rec("6_shortcut", "esoph_hi_vox neg-only: NG-tube report vs not", 0, neg.sum(), auc(T.rep_ngtube[neg], T.esoph_hi_vox[neg]))
# unclear class
rec("6_label", "unclear (excluded) mean E3 percentile among train neg", 0, len(U), np.mean(np.searchsorted(np.sort(S[neg]), U.E3_late.values) / neg.sum()))
rec("6_label", "unclear: frac scoring above 90%spec threshold", 0, len(U), (U.E3_late.values >= np.quantile(S[neg], .9)).mean())
# 7 confounders
rec("7_conf", "age-only AUROC", pos.sum(), neg.sum(), auc(y, T.Age))
rec("7_conf", "sex-only AUROC (female=1)", pos.sum(), neg.sum(), auc(y, (T.Gender == "Female").astype(float)))
T["agebin"] = (T.Age // 5).clip(upper=19)
w = np.ones(len(T)); pb = T[pos].agebin.value_counts(normalize=True); nb = T[neg].agebin.value_counts(normalize=True)
w[neg] = T.agebin[neg].map(pb / nb).fillna(0).values
for m in MODES:
    rec("7_conf", f"{m}: age-matched AUROC (neg reweighted to pos age dist)", pos.sum(), neg.sum(), roc_auc_score(y, T[m], sample_weight=w))
T["phase2"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
for v in T.phase2.unique():
    mm = (T.phase2 == v).values
    if y[mm].sum() >= 15: rec("7_conf", f"phase={v}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
for k, nm in [("y_asc", "腹水"), ("y_fa", "腹腔游離氣體"), ("y_app", "闌尾炎")]:
    rec("7_conf", f"{nm} label as score", pos.sum(), neg.sum(), auc(y, T[k].fillna(0)))
    mm = (T[k] == 1).values; rec("7_conf", f"E3 AUROC within {nm}-positive", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
# 8 handcrafted features
FC = [c for c in ft.columns if c not in ("study_id", "err", "has_ts")]
has = T.sb_ml.notna().values if "sb_ml" in T else np.zeros(len(T), bool)
rec("0_prev", "train cases with bowel features", y[has].sum(), (1 - y[has]).sum(), has.mean())
if has.sum():
    rec("8_feat_single", "E3_late on the feature subset", y[has].sum(), (1 - y[has]).sum(), auc(y[has], S[has]))
    for f in FC:
        rec("8_feat_single", f, y[has].sum(), (1 - y[has]).sum(), auc(y[has], T[f][has]),
            f"pos median {T[f][pos & has].median():.3g} / neg median {T[f][neg & has].median():.3g}; corr with E3 (neg) {np.corrcoef(np.nan_to_num(T[f][neg & has]), S[neg & has])[0, 1]:.2f}")
T["age"] = T.Age.fillna(T.Age.median()); T["female"] = (T.Gender == "Female").astype(float)
def Xf(d, med=None):
    x = d[FC].astype(float).copy()
    for c in FC:
        if c.endswith(("_ml", "_vox", "_n3", "_max", "_ncc")) and not c.endswith("diam_max"): x[c] = np.log1p(x[c].clip(lower=0))
    return x.fillna(med if med is not None else x.median()).fillna(0)
def mk(model):
    return (make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")) if model == "lr"
            else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))
def oof_fit(X, mask, model="lr"):
    p = np.full(len(T), np.nan); X = np.asarray(X, float)
    for k in range(5):
        tr, te = (T.fold != k).values & mask, (T.fold == k).values & mask
        m = mk(model).fit(X[tr], y[tr]); p[te] = m.decision_function(X[te]) if model == "lr" else m.predict_proba(X[te])[:, 1]
    return p
STK = {}
if has.sum():
    XF = Xf(T); FMED = XF.median()
    STK["p_e3_only_lr"] = oof_fit(T[["E3_late"]], has)
    STK["p_feat_lr"] = oof_fit(XF, has)
    STK["p_feat_gbm"] = oof_fit(XF, has, "gbm")
    STK["p_demo"] = oof_fit(T[["age", "female"]], has)
    STK["p_e3_demo"] = oof_fit(T[["E3_late", "age", "female"]], has)
    STK["p_windows_lr"] = oof_fit(T[[c for c in MODES if c.startswith(("E0", "E1"))]], has)
    STK["p_stack_e3_lr"] = oof_fit(pd.concat([XF, T[["E3_late"]]], axis=1), has)
    STK["p_stack_e3_gbm"] = oof_fit(pd.concat([XF, T[["E3_late"]]], axis=1), has, "gbm")
    STK["p_stack_e0_gbm"] = oof_fit(pd.concat([XF, T[["E0"]]], axis=1), has, "gbm")
    STK["p_stack_e3_demo_gbm"] = oof_fit(pd.concat([XF, T[["E3_late", "age", "female"]]], axis=1), has, "gbm")
    if "E3_late_st" in T:
        STK["p_stack_e3st_gbm"] = oof_fit(pd.concat([XF, T[["E3_late_st"]]], axis=1), has, "gbm")
    for c, s in STK.items():
        T[c] = s
        rec("8_stack_oof", c, y[has].sum(), (1 - y[has]).sum(), auc(y[has], s[has]),
            f"age-matched {roc_auc_score(y[has], s[has], sample_weight=w[has]):.4f}; sens@90spec {(s[pos & has] >= np.quantile(s[neg & has], .9)).mean():.3f}")
    # paired bootstrap of stack - E3 on train OOF
    rng = np.random.default_rng(0); ii = np.where(has)[0]; dd = []
    for _ in range(300):
        b = rng.choice(ii, len(ii)); dd.append(auc(y[b], T.p_stack_e3_gbm.values[b]) - auc(y[b], S[b]))
    rec("8_stack_oof", "delta stack_e3_gbm - E3_late (95% CI bootstrap)", y[has].sum(), (1 - y[has]).sum(), np.mean(dd), f"[{np.percentile(dd, 2.5):.4f}, {np.percentile(dd, 97.5):.4f}]")
# 9 mask quality
if has.sum():
    for f in ["sb_ml", "sb_diam_p99", "sb_dil25_frac", "unl_fluid_adjsb_ml", "unl_fluid_ge25_ml", "unl_gas_adjsb_ml", "unl_gasnearfluid_ml"]:
        if f not in T: continue
        qs = T[f][has].quantile([.33, .67]).values
        for nm, mm in [("low", T[f] <= qs[0]), ("mid", (T[f] > qs[0]) & (T[f] <= qs[1])), ("high", T[f] > qs[1])]:
            mm = mm.values & has; rec("9_mask_tertile", f"{f} tertile {nm}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
# 10 errors at 90% spec
thr = np.quantile(S[neg], 0.90); pred = S >= thr
fn = pos & ~pred; fp = neg & pred; tp = pos & pred; tn = neg & ~pred
rec("10_err", "sensitivity@90%spec E3_late", pos.sum(), neg.sum(), tp.sum() / pos.sum())
rec("10_err", "sensitivity@90%spec E0", pos.sum(), neg.sum(), (T.E0.values[pos] >= np.quantile(T.E0.values[neg], .9)).mean())
dil_thr = T.sb_dil30_ml[neg & has].quantile(.9) if has.sum() else np.inf
def cat_fn(r):
    if r.sbo_type in ("LBO", "both") or r.kw_lbo: return "A 大腸阻塞（LBO）"
    if r.kw_gastric_outlet: return "B 胃出口／十二指腸／輸入環（biliopancreatic limb）阻塞"
    if r.kw_ileus or r.kw_no_transition: return "C 報告提到腸麻痺或無明確轉折點（ileus 混淆）"
    if r.kw_partial or r.kw_resolving: return "D 部分／早期／低度或緩解中"
    if r.sbo_cat == "probably_yes" or r.kw_uncertain: return "E 報告用語不確定（probably_yes）"
    if has_r(r) and r.sb_dil30_ml < 1: return "F 遮罩內無 >30 mm 小腸段（擴張未進遮罩或擴張輕微）"
    return "G 典型小腸阻塞仍漏判"
def has_r(r): return r.sb_ml == r.sb_ml
def cat_fp(r):
    if r.rep_aff_obstruct: return "a 報告有肯定的阻塞句（疑似漏標）"
    if r.rep_aff_ileus: return "b 腸麻痺／假性阻塞"
    if r.rep_aff_sb_dilated or r.rep_aff_colon_dilated: return "c 報告寫腸道擴張但未判定阻塞"
    if r.kw_hernia: return "d 疝氣（無阻塞）"
    if r.rep_gastroenteritis or r.rep_bowel_wall: return "e 腸炎／腸壁增厚／缺血"
    if r.rep_postop or r.rep_ostomy: return "f 術後／造口變化"
    if r.rep_ascites: return "g 腹水"
    return "h 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
basep = T[pos].apply(cat_fn, axis=1); basen = T[neg].apply(cat_fp, axis=1)
for k, v in T.fn_cat[fn].value_counts().items():
    rec("10_err_FN", k, int(v), 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(basep == k).mean():.3f}; miss rate in category={v / (basep == k).sum():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    rec("10_err_FP", k, 0, int(v), v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(basen == k).mean():.3f}; FP rate in category={v / (basen == k).sum():.3f}")
for k in basep.unique():
    mm = (basep == k).reindex(T.index, fill_value=False).values
    sub("10_err_FNcat_auc", f"pos category {k}", mm)
for k in basen.unique():
    mm = (basen == k).reindex(T.index, fill_value=False).values
    negsub("10_err_FPcat_auc", f"neg category {k}", mm)
if has.sum():
    for f in ["sb_ml", "sb_diam_max", "sb_diam_p99", "sb_dil25_ml", "sb_dil30_ml", "sb_fluid_ml", "sb_gas_ml", "sb_dense_ml", "sb_afl_n3", "co_diam_p99", "co_gas_ml",
              "st_ml", "unl_fluid_adjsb_ml", "unl_fluid_ge25_ml", "unl_gas_adjsb_ml", "esoph_hi_vox"]:
        if f in T:
            rec("10_err_feat", f"{f} median: TP / FN / FP / TN", 0, 0, np.nan,
                f"{T[f][tp].median():.3g} / {T[f][fn].median():.3g} / {T[f][fp].median():.3g} / {T[f][tn].median():.3g}")
    rec("10_err_feat", "FN with sb_dil30_ml==0 (no >30 mm segment inside SB mask)", fn.sum(), 0, (T.sb_dil30_ml[fn & has] < 0.03).mean(),
        f"TP {(T.sb_dil30_ml[tp & has] < .03).mean():.3f}; FP {(T.sb_dil30_ml[fp & has] < .03).mean():.3f}; TN {(T.sb_dil30_ml[tn & has] < .03).mean():.3f}")
    if "p_stack_e3_gbm" in T:
        ps = T.p_stack_e3_gbm.values
        rec("10_err", "sensitivity@90%spec stack_e3_gbm (bowel feats)", y[has].sum(), (1 - y[has]).sum(), (ps[pos & has] >= np.quantile(ps[neg & has], .9)).mean())
        th2 = np.quantile(ps[neg & has], .9); fn2 = pos & has & (ps < th2)
        for k, v in T.fn_cat[fn & has].value_counts().items():
            rec("10_err_rescued", f"FN cat {k}: frac rescued by stack", int(v), 0, ((T.fn_cat == k).values & fn & has & ~fn2).sum() / v)
# 11 ceiling / label-noise cost
susp_neg = neg & ((T.rep_aff_obstruct == 1) | (T.rep_aff_ileus == 1)).values
susp_pos = pos & ((T.kw_ileus == 1) | (T.kw_no_transition == 1)).values
for nm, mm in [("drop neg with affirmative obstruct/ileus sentence", ~susp_neg), ("drop pos with ileus/no-transition wording", ~susp_pos),
               ("drop both", ~(susp_neg | susp_pos)), ("drop both + probably_yes", ~(susp_neg | susp_pos | (pos & (T.sbo_cat == "probably_yes").values)))]:
    rec("11_ceiling", f"E3_late {nm}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
    if "p_stack_e3_gbm" in T: rec("11_ceiling", f"stack_e3_gbm {nm}", y[mm & has].sum(), (1 - y[mm & has]).sum(), auc(y[mm & has], T.p_stack_e3_gbm.values[mm & has]))
mm = ~(susp_neg | susp_pos) & ((T.sbo_type == "SBO").values | neg)
rec("11_ceiling", "E3_late clean, SBO-type pos only", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
keepc = (["study_id", "fold", "y_sbo", "sbo_cat", "sbo_type", "kw_partial", "kw_highgrade", "kw_ileus", "kw_no_transition", "rep_postop", "rep_ngtube", "phase2", "Age", "E0", "E3_late"]
         + [c for c in ["E3_late_st", "p_feat_gbm", "p_stack_e3_gbm", "sb_ml", "sb_diam_p99", "sb_dil30_ml", "unl_fluid_adjsb_ml"] if c in T] + ["fn_cat", "fp_cat"])
T[keepc].to_csv(A + "per_case_train_oof.csv", index=False)
T[T.fn_cat != ""][["study_id", "fn_cat", "E3_late"]].to_csv(A + "errors_fn_at_spec90.csv", index=False)
T[T.fp_cat != ""][["study_id", "fp_cat", "E3_late"]].to_csv(A + "errors_fp_at_spec90.csv", index=False)
# VAL descriptive (expB probabilities for E0/E3_late; stack fitted on full train, applied once)
V = df[df.split == "val"].copy(); V = V[V.y_sbo.notna()]
V = V.merge(pd.read_csv(A + "expB_val_scores.csv"), on="study_id", how="left")
V = V.merge(pd.read_csv(A + "scores_val_descriptive.csv"), on="study_id", how="left")
yv = V.y_sbo.values.astype(int); vp = yv == 1
vrows = []
def vsub(model, sname, mk_):
    s = V[model].values; mk_ = np.asarray(mk_, bool)
    vrows.append(dict(model=model, subset=sname, auc=round(auc(yv[mk_], s[mk_]), 4), n_pos=int(yv[mk_].sum()), n_neg=int((1 - yv[mk_]).sum())))
for nm in ["E0_expB", "E3_late_expB", "E0", "E3_late"] + [c for c in ["E3_late_st"] if c in V]:
    vsub(nm, "all", np.ones(len(V), bool))
    vsub(nm, "type=SBO pos vs all neg", (vp & (V.sbo_type == "SBO").values) | ~vp)
    vsub(nm, "LBO/both pos vs all neg", (vp & V.sbo_type.isin(["LBO", "both"]).values) | ~vp)
    vsub(nm, "partial/low-grade pos vs all neg", (vp & (V.kw_partial == 1).values) | ~vp)
    vsub(nm, "all pos vs neg with affirmative obstruct/ileus/dilated sentence", vp | (~vp & (V[["rep_aff_obstruct", "rep_aff_ileus", "rep_aff_sb_dilated"]].sum(1) > 0).values))
    vsub(nm, "drop suspicious labels", ~((~vp & ((V.rep_aff_obstruct == 1) | (V.rep_aff_ileus == 1)).values) | (vp & ((V.kw_ileus == 1) | (V.kw_no_transition == 1)).values)))
if has.sum():
    hv = V.sb_ml.notna().values
    XT = pd.concat([XF, T[["E3_late"]]], axis=1)[has]
    m = mk("gbm").fit(np.asarray(XT, float), y[has])
    xv = Xf(V[hv], FMED); xv["E3_late"] = V.E3_late.values[hv]
    V.loc[hv, "stack_e3_gbm"] = m.predict_proba(np.asarray(xv, float))[:, 1]
    m2 = mk("gbm").fit(np.asarray(XF[has], float), y[has]); V.loc[hv, "feat_gbm"] = m2.predict_proba(np.asarray(Xf(V[hv], FMED), float))[:, 1]
    for nm in ["E3_late", "feat_gbm", "stack_e3_gbm"]:
        s = V[nm].values
        vrows.append(dict(model=nm, subset="val with bowel feats", auc=round(auc(yv[hv], s[hv]), 4), n_pos=int(yv[hv].sum()), n_neg=int((1 - yv[hv]).sum())))
    rng = np.random.default_rng(1); ii = np.where(hv)[0]; dd = []
    for _ in range(500):
        b = rng.choice(ii, len(ii)); dd.append(auc(yv[b], V.stack_e3_gbm.values[b]) - auc(yv[b], V.E3_late.values[b]))
    vrows.append(dict(model="stack_e3_gbm - E3_late", subset=f"val delta, 95% CI [{np.percentile(dd, 2.5):.4f}, {np.percentile(dd, 97.5):.4f}]", auc=round(np.mean(dd), 4), n_pos=int(yv[hv].sum()), n_neg=int((1 - yv[hv]).sum())))
    V[["study_id", "E0", "E3_late", "stack_e3_gbm"]].to_csv(A + "scores_val_descriptive_stack.csv", index=False)
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 160); pd.set_option("display.max_rows", 600)
print(res.to_string()); print(pd.DataFrame(vrows).to_string())
