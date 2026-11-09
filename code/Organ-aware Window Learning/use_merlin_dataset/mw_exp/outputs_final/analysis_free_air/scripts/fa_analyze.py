"""Free-air error analysis on TRAIN OOF (val descriptive only, no tuning on val, test never read)."""
import numpy as np, pandas as pd, json, warnings, os
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_free_air/"
lab = pd.read_csv(A + "labels_trval.csv")
air = pd.concat([pd.read_csv(A + f) for f in ("air_a6.csv", "air_ct.csv") if os.path.exists(A + f)]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase"]]
df = lab.merge(air, on="study_id", how="left").merge(meta, on="study_id", how="left")
oof = pd.read_csv(A + "oof_train.csv")
T = df[df.split == "train"].merge(oof, on="study_id"); T = T[T.y_fa.notna()].reset_index(drop=True)
y = T.y_fa.values.astype(int); S = T.E3_late.values
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
out = []
def rec(sec, name, n1, n0, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
MODES = [c for c in ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"] if c in T]
for m in MODES: rec("1_overall_oof", m, y.sum(), (1 - y).sum(), auc(y, T[m]))
neg = y == 0; pos = y == 1
T["postop"] = (T.fa_postop == "yes") | (T.kw_postop == 1) | (T.kw_expected == 1)
T["nonpostop"] = ~T.postop
negS = neg & (T.rep_surgery == 1).values; negN = neg & (T.rep_surgery == 0).values
rec("0_prev", "pos frac postop (label detail=yes OR postop/expected kw)", pos.sum(), 0, T.postop[pos].mean())
rec("0_prev", "pos frac label detail postoperative=yes", pos.sum(), 0, (T.fa_postop[pos] == "yes").mean())
rec("0_prev", "neg frac report mentions surgery/drain/anastomosis", 0, neg.sum(), T.rep_surgery[neg].mean())
rec("0_prev", "pos frac report mentions surgery/drain/anastomosis", pos.sum(), 0, T.rep_surgery[pos].mean())
def sub(sec, name, pm, negmask=neg, s=S):
    mm = (pm & pos) | negmask
    rec(sec, name, (pm & pos).sum(), negmask.sum(), auc(y[mm], s[mm]),
        f"frac_of_pos={(pm & pos).sum()/pos.sum():.3f}; vs surgical neg={auc(y[(pm&pos)|negS], s[(pm&pos)|negS]):.3f}; vs non-surgical neg={auc(y[(pm&pos)|negN], s[(pm&pos)|negN]):.3f}")
# 2 post-op
PO = T.postop.values
sub("2_postop", "pos postop", PO); sub("2_postop", "pos NON-postop", ~PO)
for v in ["yes", "no", "not_stated"]: sub("2_postop", f"label postoperative={v}", (T.fa_postop == v).values)
for k in ["kw_expected", "kw_drain", "kw_perf"]: sub("2_postop", k + "=1", (T[k] == 1).values); sub("2_postop", k + "=0", (T[k] == 0).values)
for m in MODES:
    rec("2_surgery_shortcut", f"{m}: neg-only AUROC surgery-in-report vs not (higher=model scores surgery)", 0, neg.sum(), auc(T.rep_surgery[neg], T[m][neg]))
    mm = (PO & pos) | negS; rec("2_postop_by_model", f"{m}: postop pos vs surgical neg", (PO & pos).sum(), negS.sum(), auc(y[mm], T[m].values[mm]))
    mm = (~PO & pos) | negN; rec("2_postop_by_model", f"{m}: NON-postop pos vs non-surgical neg", (~PO & pos).sum(), negN.sum(), auc(y[mm], T[m].values[mm]))
    mm = (~PO & pos) | neg; rec("2_postop_by_model", f"{m}: NON-postop pos vs all neg", (~PO & pos).sum(), neg.sum(), auc(y[mm], T[m].values[mm]))
rec("2_surgery_shortcut", "rep_surgery as a score (all)", pos.sum(), neg.sum(), auc(y, T.rep_surgery))
if "metal_vox" in T:
    rec("2_surgery_shortcut", "metal_vox (>1500HU, non-bone) as score", pos.sum(), neg.sum(), auc(y, T.metal_vox))
    rec("2_surgery_shortcut", "metal_vox neg-only surgery vs not", 0, neg.sum(), auc(T.rep_surgery[neg], T.metal_vox[neg]))
# 3 amount / 4 location / 6 label
sub("3_amount", "small/trace/few (quote)", ((T.kw_small == 1) & (T.kw_large == 0)).values)
sub("3_amount", "large/moderate (quote)", (T.kw_large == 1).values)
sub("3_amount", "amount not stated", ((T.kw_small == 0) & (T.kw_large == 0)).values)
sub("3_amount", "decreasing/interval", (T.kw_decreasing == 1).values)
for k in ["kw_retroperi", "kw_retro_only", "kw_subdiaph", "kw_pelvic", "kw_wall", "kw_perf"]: sub("4_location", k + "=1", (T[k] == 1).values)
sub("6_label", "category probably_yes", (T.fa_cat == "probably_yes").values); sub("6_label", "category yes", (T.fa_cat == "yes").values)
for k in ["kw_uncertain", "kw_history", "kw_pneumatosis", "kw_pneumobilia"]: sub("6_label", k + "=1", (T[k] == 1).values)
rec("6_label", "neg with affirmative free-air sentence in report (possible missed pos)", 0, (neg & (T.rep_affirm_fa == 1).values).sum(), (neg & (T.rep_affirm_fa == 1).values).sum() / neg.sum())
mm = (neg & (T.rep_affirm_fa == 1).values)
rec("6_label", "mean E3 percentile among neg of affirm-neg", 0, mm.sum(), pd.Series(S[neg]).rank(pct=True).values[(T.rep_affirm_fa[neg] == 1).values].mean())
rec("6_label", "neg category not_mentioned frac", 0, neg.sum(), (T.fa_cat[neg] == "not_mentioned").mean())
for c in ["no", "not_mentioned", "probably_no"]:
    mm = pos | (neg & (T.fa_cat == c).values); rec("6_label", f"pos vs neg cat={c}", pos.sum(), (neg & (T.fa_cat == c).values).sum(), auc(y[mm], S[mm]))
# 7 confounders
rec("7_conf", "age-only AUROC", pos.sum(), neg.sum(), auc(y, T.Age))
rec("7_conf", "age-only AUROC (younger=pos)", pos.sum(), neg.sum(), auc(y, -T.Age))
T["agebin"] = (T.Age // 5).clip(upper=19)
w = np.ones(len(T)); pb = T[pos].agebin.value_counts(normalize=True); nb = T[neg].agebin.value_counts(normalize=True)
w[neg] = T.agebin[neg].map(pb / nb).fillna(0).values
for m in MODES:
    rec("7_conf", f"{m}: age-matched AUROC (neg reweighted to pos age dist)", pos.sum(), neg.sum(), roc_auc_score(y, T[m], sample_weight=w))
for k, nm in [("y_asc", "腹水"), ("y_sbo", "腸阻塞"), ("y_app", "闌尾炎")]:
    rec("7_conf", f"{nm} label as score", pos.sum(), neg.sum(), auc(y, T[k].fillna(0)))
    mm = (T[k] != 1).values; rec("7_conf", f"E3 AUROC within {nm}-negative", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
    mm = (T[k] == 1).values; rec("7_conf", f"E3 AUROC within {nm}-positive", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
T["phase2"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
for v in T.phase2.unique():
    mm = (T.phase2 == v).values
    if y[mm].sum() >= 15: rec("7_conf", f"phase={v}", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
# 5/8 air features
AF = [c for c in T.columns if c.startswith("air") or c in ("metal_vox", "metal_ncc", "gut_vox", "lung_vox")]
has = T.air900_total.notna().values if "air900_total" in T else np.zeros(len(T), bool)
rec("0_prev", "train cases with air features", y[has].sum(), (1 - y[has]).sum(), has.mean())
for f in AF:
    rec("5_air_single", f, y[has].sum(), (1 - y[has]).sum(), auc(y[has], T[f][has]),
        f"pos median {T[f][pos & has].median():.0f} / neg median {T[f][neg & has].median():.0f}; non-postop pos vs non-surg neg {auc(y[((~PO&pos)|negN)&has], T[f][((~PO&pos)|negN)&has]):.3f}")
T["age"] = T.Age.fillna(T.Age.median()); T["female"] = (T.Gender == "Female").astype(float)
def X_air(d):
    x = d[AF].copy()
    for c in AF: x[c] = np.log1p(x[c].clip(lower=0))
    return x.fillna(x.median())
def oof_lr(X, mask):
    p = np.full(len(T), np.nan)
    for k in range(5):
        tr, te = (T.fold != k).values & mask, (T.fold == k).values & mask
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
        mdl.fit(X[tr], y[tr]); p[te] = mdl.decision_function(X[te])
    return p
if has.sum():
    XA = X_air(T)
    T["p_air"] = oof_lr(XA, has)
    T["p_demo"] = oof_lr(T[["age", "female"]], has)
    T["p_e3_demo"] = oof_lr(T[["E3_late", "age", "female"]], has)
    T["p_stack_e0"] = oof_lr(pd.concat([XA, T[["E0"]]], axis=1), has)
    T["p_stack_e3"] = oof_lr(pd.concat([XA, T[["E3_late"]]], axis=1), has)
    T["p_stack_e3_demo"] = oof_lr(pd.concat([XA, T[["E3_late", "age", "female"]]], axis=1), has)
    T["p_e3_only_lr"] = oof_lr(T[["E3_late"]], has)
    allw = [c for c in MODES if c != "E3_late"]
    T["p_stack_windows"] = oof_lr(T[allw], has)
    for c in ["p_e3_only_lr", "p_air", "p_demo", "p_e3_demo", "p_stack_e0", "p_stack_e3", "p_stack_e3_demo", "p_stack_windows"]:
        s = T[c].values
        rec("8_stack_oof", c, y[has].sum(), (1 - y[has]).sum(), auc(y[has], s[has]),
            f"NON-postop pos vs non-surg neg: {auc(y[((~PO&pos)|negN)&has], s[((~PO&pos)|negN)&has]):.3f}; postop pos vs surg neg: {auc(y[((PO&pos)|negS)&has], s[((PO&pos)|negS)&has]):.3f}")
# 9 errors at 90% spec
thr = np.quantile(S[neg], 0.90); pred = S >= thr
fn = pos & ~pred; fp = neg & pred; tp = pos & pred
rec("9_err", "sensitivity@90%spec E3_late", pos.sum(), neg.sum(), tp.sum() / pos.sum())
if "p_stack_e3" in T:
    ps = T.p_stack_e3.values; m2 = has
    rec("9_err", "sensitivity@90%spec stack_e3 (air feats)", y[m2].sum(), (1 - y[m2]).sum(), (ps[pos & m2] >= np.nanquantile(ps[neg & m2], .9)).mean())
q75 = T.air900_extra_deep[neg].quantile(.9) if "air900_extra_deep" in T else np.inf
def cat_fn(r):
    if r.kw_pneumatosis or r.kw_pneumobilia: return "B 報告同時提及腸壁氣腫／膽道積氣（可能非真游離氣體）"
    if r.kw_retro_only or (r.kw_wall and not r.kw_subdiaph): return "C 後腹腔或腹壁／皮下氣體"
    if not r.postop: return "F 非術後（自發性／穿孔／原因未述）"
    if r.fa_cat == "probably_yes": return "A 術後、報告用語為可能（probably_yes）"
    if r.kw_small and not r.kw_large: return "D 術後少量（trace/少許/小氣泡）"
    return "E 術後中大量／未述量"
def cat_fp(r):
    if r.rep_affirm_fa: return "a 報告其實有肯定的游離氣體句（疑似漏標）"
    if r.kw_pneumatosis or r.kw_pneumobilia: return "b 腸壁氣腫／膽道積氣／門靜脈氣體"
    if r.y_sbo == 1: return "c 腸阻塞（腸道大量積氣）"
    if r.rep_surgery: return "d 術後／引流管／吻合等手術變化（無游離氣體）"
    if r.y_asc == 1: return "e 腹水"
    return "f 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
basep = T[pos].apply(cat_fn, axis=1); basen = T[neg].apply(cat_fp, axis=1)
for k, v in T.fn_cat[fn].value_counts().items():
    rec("9_err_FN", k, int(v), 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(basep==k).mean():.3f}; miss rate in category={v/(basep==k).sum():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    rec("9_err_FP", k, 0, int(v), v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(basen==k).mean():.3f}")
if has.sum():
    for f in ["air900_extra_deep", "air900_deep_small", "air900_ring", "air900_inlumen", "air900_antliver", "metal_vox", "gut_vox"]:
        rec("9_err_air", f"{f} median: TP / FN / FP / TN", 0, 0, np.nan,
            f"{T[f][tp].median():.0f} / {T[f][fn].median():.0f} / {T[f][fp].median():.0f} / {T[f][neg & ~pred].median():.0f}")
    rec("9_err_air", "FN with air900_deep_small==0 (no detectable extraluminal air at 3mm)", fn.sum(), 0, (T.air900_deep_small[fn & has] == 0).mean(),
        f"TP: {(T.air900_deep_small[tp & has] == 0).mean():.3f}; TN: {(T.air900_deep_small[neg & ~pred & has] == 0).mean():.3f}")
    rec("9_err_air", "FP with air900_ring>0 (gas just outside gut mask)", 0, fp.sum(), (T.air900_ring[fp & has] > 0).mean(), f"TN: {(T.air900_ring[neg & ~pred & has] > 0).mean():.3f}")
# 10 ceiling
susp_pos = pos & ((T.fa_cat == "probably_yes") | (T.kw_pneumatosis == 1) | (T.kw_pneumobilia == 1) | (T.kw_retro_only == 1)).values
susp_neg = neg & (T.rep_affirm_fa == 1).values
mm = ~(susp_pos | susp_neg)
rec("10_ceiling", "E3_late after removing suspicious labels", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]), f"removed pos {susp_pos.sum()} neg {susp_neg.sum()}")
mm2 = mm & ((PO & pos) | negS | (~PO & pos) | negN)
mm3 = mm & ((~PO & pos) | negN)
rec("10_ceiling", "E3_late clean non-postop pos vs non-surgical neg", y[mm3].sum(), (1 - y[mm3]).sum(), auc(y[mm3], S[mm3]))
if "p_stack_e3" in T:
    rec("10_ceiling", "stack_e3 clean labels", y[mm & has].sum(), (1 - y[mm & has]).sum(), auc(y[mm & has], T.p_stack_e3[mm & has]))
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
keepc = ["study_id", "fold", "y_fa", "fa_cat", "fa_postop", "postop", "rep_surgery", "kw_small", "kw_large", "phase2", "Age",
         "E0", "E3_late"] + [c for c in ["p_air", "p_stack_e3", "air900_extra_deep", "air900_deep_small", "air900_ring", "metal_vox"] if c in T] + ["fn_cat", "fp_cat"]
T[keepc].to_csv(A + "per_case_train_oof.csv", index=False)
# VAL descriptive (expB probabilities, subgroups only; stack fitted on train OOF table, applied once)
V = df[df.split == "val"].copy(); V = V[V.y_fa.notna()]
for nm in ["E0", "E3_late"]:
    z = np.load(R + f"results/expB/{nm}.npz", allow_pickle=True); fi = list(z["findings"]).index("free_air")
    spl = pd.Series(z["split"], index=z["ids"]); pv = pd.Series(z["prob"].mean(0)[:, fi], index=z["ids"])[spl == "val"]
    V[nm] = V.study_id.map(pv)
V = V[V.E3_late.notna()]; yv = V.y_fa.values.astype(int)
V["postop"] = (V.fa_postop == "yes") | (V.kw_postop == 1) | (V.kw_expected == 1)
vp, vn = yv == 1, yv == 0; vnS = vn & (V.rep_surgery == 1).values; vnN = vn & (V.rep_surgery == 0).values; VPO = V.postop.values
vrows = []
for nm in ["E0", "E3_late"]:
    s = V[nm].values
    for sn, mk in [("all", np.ones(len(V), bool)), ("postop pos vs all neg", (VPO & vp) | vn), ("NON-postop pos vs all neg", (~VPO & vp) | vn),
                   ("NON-postop pos vs non-surgical neg", (~VPO & vp) | vnN), ("postop pos vs surgical neg", (VPO & vp) | vnS)]:
        vrows.append(dict(model=nm, subset=sn, auc=round(auc(yv[mk], s[mk]), 4), n_pos=int(yv[mk].sum()), n_neg=int((1 - yv[mk]).sum())))
if has.sum():
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(pd.concat([XA, T[["E3_late"]]], axis=1)[has], y[has])
    V2 = df[df.split == "val"].merge(pd.read_csv(A + "scores_val_descriptive.csv"), on="study_id"); V2 = V2[V2.y_fa.notna() & V2.air900_total.notna()]
    xv = V2[AF].copy()
    for c in AF: xv[c] = np.log1p(xv[c].clip(lower=0))
    xv = xv.fillna(XA.median()); xv["E3_late"] = V2.E3_late.values
    ps = mdl.decision_function(xv); yv2 = V2.y_fa.values.astype(int)
    po2 = ((V2.fa_postop == "yes") | (V2.kw_postop == 1) | (V2.kw_expected == 1)).values; nn2 = (yv2 == 0) & (V2.rep_surgery == 0).values
    mk = (~po2 & (yv2 == 1)) | nn2
    for nm, sc in [("E3_late (this recipe)", V2.E3_late.values), ("stack air+E3_late (fit on train)", ps)]:
        vrows.append(dict(model=nm, subset="all (val with air feats)", auc=round(auc(yv2, sc), 4), n_pos=int(yv2.sum()), n_neg=int((1 - yv2).sum())))
        vrows.append(dict(model=nm, subset="NON-postop pos vs non-surgical neg", auc=round(auc(yv2[mk], sc[mk]), 4), n_pos=int(yv2[mk].sum()), n_neg=int((1 - yv2[mk]).sum())))
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 400)
print(res.to_string()); print(pd.DataFrame(vrows).to_string())
