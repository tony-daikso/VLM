import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_gallstones/"
lab = pd.read_csv(A + "labels_trval.csv")
hu = pd.concat([pd.read_csv(A + "hu_a6.csv"), pd.read_csv(A + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "slicethickness"]]
oof = pd.read_csv(A + "oof_train.csv")
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
T = df[df.split == "train"].merge(oof, on="study_id")
T = T[T.y_gs.notna()].reset_index(drop=True)
y = T.y_gs.values.astype(int)
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
out = []
def rec(sec, name, n1, n0, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=n1, n_neg=n0, value=round(float(val), 4) if val == val else np.nan, note=note))
MODES = ["E0", "E3_late", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]
for m in MODES:
    rec("1_overall_oof", m, y.sum(), (1 - y).sum(), auc(y, T[m]))
S = T["E3_late"].values
# features
T["gb_present"] = (T.gb_vox >= 100).astype(int)
T["vis100"] = (T.gb_maxblob100 >= 2).astype(int)   # >=2 connected voxels >100 HU (>=54 mm3)
T["vis_any100"] = (T.gb_n100 >= 1).astype(int)
T["phase2"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
T["aorta_hi"] = np.where(T.aorta_hu_med > 150, "aorta>150HU", np.where(T.aorta_hu_med < 70, "aorta<70HU", "aorta70-150"))
T["agebin"] = pd.cut(T.Age, [0, 40, 60, 75, 120], labels=["<40", "40-60", "60-75", ">=75"])
neg_all = y == 0
neg_gb = (y == 0) & (T.y_chole != 1)
# 2. negatives composition
rec("2_neg_mix", "AUROC: pos vs all neg", y.sum(), neg_all.sum(), auc(y, S))
m = (y == 1) | neg_gb
rec("2_neg_mix", "AUROC: pos vs neg with GB (exclude cholecystectomy)", y.sum(), neg_gb.sum(), auc(y[m], S[m]))
m = (y == 1) | (neg_gb & (T.gb_present == 1))
rec("2_neg_mix", "AUROC: pos vs neg with GB mask>=100vox", y.sum(), m.sum() - y.sum(), auc(y[m], S[m]))
rec("2_neg_mix", "frac of neg that are cholecystectomy", 0, neg_all.sum(), (T.y_chole[neg_all] == 1).mean())
rec("2_neg_mix", "contradiction: gallstones+ & cholecystectomy+", int(((y == 1) & (T.y_chole == 1)).sum()), 0, ((y == 1) & (T.y_chole == 1)).sum())
# 3. positive subgroups vs fixed negative set (GB-present negatives) and vs all negatives
def pos_sub(sec, name, pm, negmask=neg_all):
    mm = (pm & (y == 1)) | negmask
    rec(sec, name, int((pm & (y == 1)).sum()), int(negmask.sum()), auc(y[mm], S[mm]),
        f"frac_of_pos={(pm & (y==1)).sum()/y.sum():.3f}; vs GB-present neg={auc(y[(pm&(y==1))|neg_gb], S[(pm&(y==1))|neg_gb]):.3f}")
pos_sub("3_pos_sub", "label calcified=yes", T.gs_calc == "yes")
pos_sub("3_pos_sub", "label calcified=no", T.gs_calc == "no")
pos_sub("3_pos_sub", "label calcified=not_stated", T.gs_calc == "not_stated")
pos_sub("3_pos_sub", "HU visible (blob>100HU >=2 vox)", T.vis100 == 1)
pos_sub("3_pos_sub", "HU not visible", T.vis100 == 0)
pos_sub("3_pos_sub", "GB mask <100 vox (missing)", T.gb_vox < 100)
pos_sub("3_pos_sub", "GB mask >=100 vox", T.gb_vox >= 100)
for k in [c for c in T.columns if c.startswith("kw_")]:
    pos_sub("3_pos_sub_kw", k + "=1", T[k] == 1)
    pos_sub("3_pos_sub_kw", k + "=0", T[k] == 0)
pos_sub("3_pos_sub", "size >=10mm", T.stone_mm >= 10)
pos_sub("3_pos_sub", "size <10mm", T.stone_mm < 10)
pos_sub("3_pos_sub", "category probably_yes", T.gs_cat == "probably_yes")
pos_sub("3_pos_sub", "category yes", T.gs_cat == "yes")
# prevalence of visibility
for g in ["vis100", "vis_any100"]:
    rec("3_visibility", f"{g} rate in pos", y.sum(), 0, T[g][y == 1].mean())
    rec("3_visibility", f"{g} rate in GB-present neg", 0, neg_gb.sum(), T[g][neg_gb & (T.gb_present == 1)].mean())
rec("3_visibility", "label calcified=yes rate in pos", y.sum(), 0, (T.gs_calc[y == 1] == "yes").mean())
# GB size
for nm, mk in [("pos", y == 1), ("neg_GB", neg_gb), ("neg_chole", (y == 0) & (T.y_chole == 1))]:
    q = T.gb_vox[mk].quantile([.1, .25, .5, .75, .9]).round(0).to_dict()
    rec("4_gb_size", f"gb_vox median {nm}", mk.sum() if nm == "pos" else 0, 0 if nm == "pos" else mk.sum(), q[.5], json.dumps(q))
    rec("4_gb_size", f"gb mask missing(<100) frac {nm}", 0, 0, (T.gb_vox[mk] < 100).mean())
qs = pd.qcut(T.gb_vox.where(T.gb_vox >= 100), 4, labels=["Q1小", "Q2", "Q3", "Q4大"])
for qn in ["Q1小", "Q2", "Q3", "Q4大"]:
    mm = (qs == qn).values & ((y == 1) | neg_gb)
    rec("4_gb_size", f"AUROC within GB-size {qn} (GB-present neg)", int(y[mm].sum()), int((1 - y[mm]).sum()), auc(y[mm], S[mm]))
# 5. phase / confounders (pos & neg both within subgroup; negatives with GB)
for col in ["phase2", "aorta_hi", "agebin", "Gender", "manufacturer"]:
    for v in T[col].dropna().unique():
        mm = (T[col] == v).values & ((y == 1) | neg_gb)
        if y[mm].sum() >= 20:
            rec("5_strata", f"{col}={v}", int(y[mm].sum()), int((1 - y[mm]).sum()), auc(y[mm], S[mm]))
# single features
feats = ["gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130", "gb_n200",
         "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med", "Age"]
for f in feats:
    a_all = auc(y, T[f].fillna(-1e9 if f.startswith("gb") else T[f].median()))
    mm = ((y == 1) | neg_gb) & (T.gb_present == 1).values
    rec("6_single_feature", f, y.sum(), neg_all.sum(), a_all, f"GB-present only: {auc(y[mm], T[f][mm]):.3f}")
rec("6_single_feature", "male", y.sum(), neg_all.sum(), auc(y, (T.Gender == "Male").astype(float)))
rec("6_single_feature", "biliary_dilation label", y.sum(), neg_all.sum(), auc(y, T.y_bdd.fillna(0)))
# 7. stacked: OOF LR on HU feats, + embedding logit (same folds)
T["female"] = (T.Gender == "Female").astype(float)
HU = ["gb_present", "gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130",
      "gb_n200", "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]
X = T[HU].copy()
for c in HU:
    if c.startswith(("gb_n", "gb_vox", "shell_n", "gb_max")): X[c] = np.log1p(X[c].clip(lower=0))
X = X.fillna(X.median()); X["gb_present"] = T.gb_present
def oof_lr(cols_df):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = T.fold != k, T.fold == k
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
        mdl.fit(cols_df[tr], y[tr]); p[te] = mdl.decision_function(cols_df[te])
    return p
T["p_hu"] = oof_lr(X)
T["p_demo"] = oof_lr(pd.DataFrame({"age": T.Age.fillna(T.Age.median()), "female": T.female}))
T["p_stack_e3"] = oof_lr(pd.concat([X, T[["E3_late"]]], axis=1))
T["p_stack_e0"] = oof_lr(pd.concat([X, T[["E0"]]], axis=1))
T["p_stack_e3_demo"] = oof_lr(pd.concat([X, T[["E3_late"]], pd.DataFrame({"age": T.Age.fillna(60), "female": T.female})], axis=1))
T["p_e3_demo"] = oof_lr(pd.concat([T[["E3_late"]], pd.DataFrame({"age": T.Age.fillna(60), "female": T.female})], axis=1))
allw = T[["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]]
T["p_stack_windows"] = oof_lr(allw)
T["p_stack_windows_hu"] = oof_lr(pd.concat([allw, X], axis=1))
for c in ["p_hu", "p_demo", "p_e3_demo", "p_stack_e0", "p_stack_e3", "p_stack_e3_demo", "p_stack_windows", "p_stack_windows_hu"]:
    mm = ((y == 1) | neg_gb).values
    rec("7_stack_oof", c, y.sum(), neg_all.sum(), auc(y, T[c]), f"GB-present neg: {auc(y[mm], T[c][mm]):.3f}")
# 8. error categories at 90% spec (E3_late OOF)
thr = np.quantile(S[y == 0], 0.90)
pred = S >= thr
fn = (y == 1) & ~pred; fp = (y == 0) & pred; tp = (y == 1) & pred
rec("8_err", "sensitivity@90%spec E3_late", y.sum(), neg_all.sum(), tp.sum() / y.sum())
rec("8_err", "sensitivity@90%spec stack_e3", y.sum(), neg_all.sum(), (T.p_stack_e3[y == 1] >= np.quantile(T.p_stack_e3[y == 0], .9)).mean())
def cat_fn(r):
    if r.y_chole == 1: return "A 標註矛盾(同時標膽囊切除)"
    if r.gb_vox < 100: return "B 膽囊遮罩缺失/過小(<100 vox)"
    if r.kw_cbd_stone and not r.kw_multiple: return "C 實為總膽管結石/非膽囊"
    if r.gs_cat == "probably_yes" or r.kw_uncertain: return "D 報告用語不確定"
    if r.kw_sludge and not r.vis100: return "E 僅膽泥/膽泥樣"
    if not r.vis100: return "F 非高密度(CT 等密度/小結石)"
    return "G 高密度可見但仍漏"
def cat_fp(r):
    if r.gb_vox < 100: return "a 膽囊遮罩缺失/過小"
    if r.vis100: return "b 膽囊內有高密度(>100HU)：可能漏標/顯影劑/壁鈣化"
    if r.kw_sludge: return "c 報告提及膽泥"
    if r.kw_cholecystitis and r.kw_distended: return "d 膽囊擴張/發炎"
    if r.y_bdd == 1: return "e 合併膽管擴張"
    return "f 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), "")
T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
for k, v in T.fn_cat[fn].value_counts().items():
    rec("8_err_FN", k, int(v), 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(T[y==1].apply(cat_fn,axis=1)==k).mean():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    rec("8_err_FP", k, 0, int(v), v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(T[y==0].apply(cat_fp,axis=1)==k).mean():.3f}")
# 9. ceiling: drop suspicious labels and see AUROC; visible-only
susp_pos = (y == 1) & ((T.y_chole == 1) | (T.kw_cbd_stone & ~T.kw_multiple.astype(bool)) | (T.gs_cat == "probably_yes"))
susp_neg = (y == 0) & (T.y_chole != 1) & (T.vis100 == 1) & (T.gb_present == 1)
rec("9_ceiling", "susp pos frac", susp_pos.sum(), 0, susp_pos.sum() / y.sum())
rec("9_ceiling", "neg with GB-blob>100HU frac (possible missed stones / contrast)", 0, susp_neg.sum(), susp_neg.sum() / neg_gb.sum())
mm = ~(susp_pos | susp_neg).values
rec("9_ceiling", "E3_late AUROC after removing suspicious labels", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
mm2 = ~susp_pos.values & ((y == 0) | (T.vis100 == 1).values)
rec("9_ceiling", "E3_late AUROC visible-pos only", y[mm2].sum(), (1 - y[mm2]).sum(), auc(y[mm2], S[mm2]))
rec("9_ceiling", "stack_e3 AUROC visible-pos only", y[mm2].sum(), (1 - y[mm2]).sum(), auc(y[mm2], T.p_stack_e3[mm2]))
fvis = T.vis100[y == 1].mean()
rec("9_ceiling", "mixture ceiling: vis*0.97+invis*AUC_invis_stack", 0, 0,
    fvis * 0.97 + (1 - fvis) * auc(y[((y==1)&(T.vis100==0)).values | (y==0)], T.p_stack_e3[((y==1)&(T.vis100==0)).values | (y==0)]))
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
T[["study_id", "fold", "y_gs", "y_chole", "gs_cat", "gs_calc", "gb_vox", "vis100", "gb_p99", "gb_n100", "aorta_hu_med", "phase2",
   "E0", "E3_late", "p_hu", "p_stack_e3", "fn_cat", "fp_cat"]].to_csv(A + "per_case_train_oof.csv", index=False)
# VAL descriptive only (no tuning): expB probs + subgroups
V = df[df.split == "val"].copy(); V = V[V.y_gs.notna()]
for nm in ["E0", "E3_late"]:
    z = np.load(R + f"results/expB/{nm}.npz", allow_pickle=True); fi = list(z["findings"]).index("gallstones")
    pv = pd.Series(z["prob"].mean(0)[:, fi], index=z["ids"]); spl = pd.Series(z["split"], index=z["ids"])
    pv = pv[spl == "val"]; V[nm] = V.study_id.map(pv)
V = V[V.E3_late.notna()]; yv = V.y_gs.values.astype(int)
vrows = []
for nm in ["E0", "E3_late"]:
    s = V[nm].values; ngb = (yv == 0) & (V.y_chole != 1).values
    vis = (V.gb_maxblob100 >= 2).values
    vrows += [dict(model=nm, subset="all", auc=auc(yv, s), n_pos=yv.sum()),
              dict(model=nm, subset="neg with GB", auc=auc(yv[(yv == 1) | ngb], s[(yv == 1) | ngb]), n_pos=yv.sum()),
              dict(model=nm, subset="pos HU-visible vs all neg", auc=auc(yv[(vis & (yv == 1)) | (yv == 0)], s[(vis & (yv == 1)) | (yv == 0)]), n_pos=int((vis & (yv == 1)).sum())),
              dict(model=nm, subset="pos HU-invisible vs all neg", auc=auc(yv[(~vis & (yv == 1)) | (yv == 0)], s[(~vis & (yv == 1)) | (yv == 0)]), n_pos=int((~vis & (yv == 1)).sum()))]
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 300)
print(res.to_string()); print(pd.DataFrame(vrows).round(4).to_string())
