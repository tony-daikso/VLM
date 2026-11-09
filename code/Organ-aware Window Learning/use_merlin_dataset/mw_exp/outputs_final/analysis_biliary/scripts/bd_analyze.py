"""Biliary dilation error analysis on TRAIN OOF (val descriptive only). Outputs results_train_oof.csv, per_case_train_oof.csv,
val_descriptive.csv in analysis_biliary/."""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_biliary/"
lab = pd.read_csv(A + "labels_trval.csv")
hu = pd.concat([pd.read_csv(A + f) for f in ("hu_ct.csv", "hu_a6.csv") if __import__("os").path.exists(A + f)]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "slicethickness"]]
oof = pd.read_csv(A + "oof_train.csv")
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
T = df[df.split == "train"].merge(oof, on="study_id")
T = T[T.y_bdd.notna()].reset_index(drop=True)
y = T.y_bdd.values.astype(int)
MODES = [c for c in oof.columns if c not in ("study_id", "fold")]
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
out = []
def rec(sec, name, n1, n0, val, note=""):
    out.append(dict(section=sec, item=name, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
print("HU features available for", T.liver_vox.notna().sum(), "of", len(T), "train")
for m in MODES:
    rec("1_overall_oof", m, y.sum(), (1 - y).sum(), auc(y, T[m]))
S = T["E3_late"].values; S0 = T["E0"].values
neg = y == 0
# ---------- derived columns
T["phase_meta"] = np.where(T.contrast == False, "non_contrast", T.phase.fillna("unknown"))
T["phase_hu"] = np.where(T.aorta_hu < 80, "平掃(aorta<80)", np.where((T.aorta_hu - T.pv_hu) >= 60, "動脈相(aorta-PV>=60)",
                 np.where(T.aorta_hu.isna(), "unknown", "門脈/延遲相")))
T["agebin"] = pd.cut(T.Age, [0, 40, 60, 75, 120], labels=["<40", "40-60", "60-75", ">=75"]).astype(str)
T["sev"] = np.where(T.kw_marked == 1, "moderate/marked", np.where(T.kw_mild == 1, "mild", "未註明"))
T["cbd_bin"] = pd.cut(T.cbd_mm, [0, 9.99, 14.99, 200], labels=["<10mm", "10-14mm", ">=15mm"]).astype(str)
T["only_prominent"] = ((T.kw_prominent == 1) & (T.qkw_prominent == 1) & ~T.bd_cat.eq("yes")).astype(int)
# ---------- 2. positive subgroups vs all negatives
def pos_sub(sec, name, pm):
    pm = np.asarray(pm, bool)
    mm = (pm & (y == 1)) | neg
    n1 = int((pm & (y == 1)).sum())
    if n1 < 15: return
    rec(sec, name, n1, neg.sum(), auc(y[mm], S[mm]), f"frac_of_pos={n1 / y.sum():.3f}; E0={auc(y[mm], S0[mm]):.3f}")
for v in ["intrahepatic", "both", "extrahepatic", "not_stated"]:
    pos_sub("2_subtype_site", f"site={v}", T.bd_site == v)
for v in ["<10mm", "10-14mm", ">=15mm"]:
    pos_sub("2_subtype_cbd", f"CBD {v}", T.cbd_bin == v)
pos_sub("2_subtype_cbd", "CBD mm not stated", T.cbd_mm.isna())
for v in ["mild", "moderate/marked", "未註明"]:
    pos_sub("2_subtype_sev", f"severity={v}", T.sev == v)
pos_sub("2_subtype_cat", "category yes", T.bd_cat == "yes")
pos_sub("2_subtype_cat", "category probably_yes", T.bd_cat == "probably_yes")
for k in ["kw_postchole", "kw_pneumobilia", "kw_stent", "kw_cbd_stone", "kw_panc_mass", "kw_pd_dilat", "kw_prominent",
          "kw_unchanged", "kw_uncertain", "kw_upper_normal", "kw_mri_rec"]:
    pos_sub("2_subtype_kw", f"{k}=1", T[k] == 1); pos_sub("2_subtype_kw", f"{k}=0", T[k] == 0)
pos_sub("2_subtype_combo", "膽囊切除+輕度+無支架/腫塊 (生理性CBD擴張樣)", (T.y_chole == 1) & (T.sev != "moderate/marked") & (T.kw_stent == 0) & (T.kw_panc_mass == 0))
pos_sub("2_subtype_combo", "支架/氣膽 (已引流)", (T.kw_stent == 1) | (T.kw_pneumobilia == 1))
pos_sub("2_subtype_combo", "阻塞型 (胰頭腫塊/總膽管結石/胰管擴張)", (T.kw_panc_mass == 1) | (T.kw_cbd_stone == 1) | (T.kw_pd_dilat == 1))
for c in ["bd_site", "sev", "cbd_bin"]:
    for v, n in T[y == 1][c].value_counts().items():
        rec("2_prevalence", f"{c}={v}", n, 0, n / y.sum())
for k in ["kw_postchole", "kw_pneumobilia", "kw_stent", "kw_cbd_stone", "kw_panc_mass", "kw_pd_dilat", "kw_prominent", "kw_unchanged", "kw_uncertain"]:
    rec("2_prevalence", k, T[k][y == 1].sum(), T[k][neg].sum(), T[k][y == 1].mean(), f"neg rate={T[k][neg].mean():.3f}")
# ---------- 3. confounders
ch = (T.y_chole == 1).values
rec("3_conf", "膽囊切除 as single predictor of 膽管擴張", y.sum(), neg.sum(), auc(y, T.y_chole.fillna(0)))
rec("3_conf", "P(膽管擴張 | 膽囊切除=1)", y[ch].sum(), (1 - y[ch]).sum(), y[ch].mean())
rec("3_conf", "P(膽管擴張 | 膽囊切除=0)", y[~ch].sum(), (1 - y[~ch]).sum(), y[~ch].mean())
for nm, mk in [("膽囊切除=1", ch), ("膽囊切除=0", ~ch)]:
    rec("3_conf_strata", f"E3_late AUROC within {nm}", y[mk].sum(), (1 - y[mk]).sum(), auc(y[mk], S[mk]), f"E0={auc(y[mk], S0[mk]):.3f}")
gbabs = (T.gb_vox < 100).values
for nm, mk in [("膽囊遮罩缺失(<100vox)", gbabs), ("膽囊遮罩存在", ~gbabs)]:
    rec("3_conf_strata", f"E3_late AUROC within {nm}", y[mk].sum(), (1 - y[mk]).sum(), auc(y[mk], S[mk]))
rec("3_conf", "mean E3 logit neg chole=1 minus neg chole=0", 0, neg.sum(), S[neg & ch].mean() - S[neg & ~ch].mean(),
    f"pos chole=1 minus pos chole=0: {S[(y==1) & ch].mean() - S[(y==1) & ~ch].mean():.3f}")
rec("3_conf", "Age single predictor", y.sum(), neg.sum(), auc(y, T.Age))
rec("3_conf", "膽結石 label single predictor", y.sum(), neg.sum(), auc(y, T.y_gs.fillna(0)))
pm = ((T.kw_panc_mass == 1) | (T.kw_pd_dilat == 1)).values
rec("3_conf", "胰臟腫塊/胰管擴張 kw single predictor", y.sum(), neg.sum(), auc(y, pm.astype(float)))
st = ((T.kw_stent == 1) | (T.kw_pneumobilia == 1)).values
rec("3_conf", "支架/氣膽 kw single predictor", y.sum(), neg.sum(), auc(y, st.astype(float)))
for v in ["<40", "40-60", "60-75", ">=75"]:
    mk = (T.agebin == v).values
    rec("3_conf_strata", f"age {v}", y[mk].sum(), (1 - y[mk]).sum(), auc(y[mk], S[mk]), f"prev={y[mk].mean():.3f}")
# age-matched AUROC (within 5-yr bins, weighted)
T["ab5"] = (T.Age // 5)
num = den = 0
for _, g in T.groupby("ab5"):
    yy = g.y_bdd.values.astype(int)
    if 0 < yy.sum() < len(yy):
        w = yy.sum() * (len(yy) - yy.sum()); num += auc(yy, g.E3_late) * w; den += w
rec("3_conf", "E3_late age-matched AUROC (5y bins)", y.sum(), neg.sum(), num / den)
num = den = 0
for _, g in T.groupby(T.y_chole.fillna(0)):
    yy = g.y_bdd.values.astype(int); w = yy.sum() * (len(yy) - yy.sum()); num += auc(yy, g.E3_late) * w; den += w
rec("3_conf", "E3_late 膽囊切除-stratified AUROC (weighted)", y.sum(), neg.sum(), num / den)
# ---------- 4. phase
for col in ["phase_meta", "phase_hu"]:
    for v in T[col].dropna().unique():
        mk = (T[col] == v).values
        if y[mk].sum() >= 15:
            rec("4_phase", f"{col}={v}", y[mk].sum(), (1 - y[mk]).sum(), auc(y[mk], S[mk]), f"E0={auc(y[mk], S0[mk]):.3f}; frac={mk.mean():.3f}")
# ---------- 6. handcrafted features: alone and stacked (OOF, same folds)
F_ = ["liv_low_n", "liv_low_frac", "liv_low_maxcc", "liv_low_ncc", "liv_fluid_n", "liv_air_n", "hil_low_n", "hil_low_frac",
      "hil_low_maxcc", "porta_fluid_n", "porta_fluid_frac", "porta_fluid_maxcc", "porta_air_n", "porta_metal_n", "porta_hu_med",
      "panc_low_n", "panc_low_frac", "phead_fluid_n", "phead_metal_n", "liver_hu_sd", "liver_vox", "panc_vox", "gb_vox", "pv_vox",
      "liver_hu", "aorta_hu", "pv_hu", "panc_hu"]
F_ = [f for f in F_ if f in T.columns]
for f in F_:
    rec("6_single_feature", f, y.sum(), neg.sum(), auc(y, T[f]), f"pos med={T[f][y==1].median():.3g} neg med={T[f][neg].median():.3g}")
def Xh(d):
    X = d[F_].copy()
    for c in F_:
        if c.endswith(("_n", "_vox", "_maxcc", "_ncc")): X[c] = np.log1p(X[c].clip(lower=0))
    X["hu_missing"] = d.liver_vox.isna().astype(float)
    return X.fillna(T[F_].median() if False else X.median())
def oof_lr(Xd, C=0.3):
    p = np.zeros(len(T))
    for k in range(5):
        trm, te = (T.fold != k).values, (T.fold == k).values
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        mdl.fit(Xd[trm], y[trm]); p[te] = mdl.decision_function(Xd[te])
    return p
XH = Xh(T)
groups = {"肝內低密度": ["liv_low_n", "liv_low_frac", "liv_low_maxcc", "liv_low_ncc", "liv_fluid_n", "liver_hu_sd"],
          "肝門低密度": ["hil_low_n", "hil_low_frac", "hil_low_maxcc"],
          "肝門外(肝十二指腸韌帶)液體密度": ["porta_fluid_n", "porta_fluid_frac", "porta_fluid_maxcc", "porta_hu_med"],
          "胰頭區/胰內低密度": ["panc_low_n", "panc_low_frac", "phead_fluid_n"],
          "氣膽/金屬(支架)": ["liv_air_n", "porta_air_n", "porta_metal_n", "phead_metal_n"]}
for g, cols in groups.items():
    cols = [c for c in cols if c in XH]
    T["p_" + g] = oof_lr(XH[cols]); rec("6_group_oof", g, y.sum(), neg.sum(), auc(y, T["p_" + g]))
T["p_hu"] = oof_lr(XH); rec("6_group_oof", "全部手工特徵", y.sum(), neg.sum(), auc(y, T.p_hu))
noart = [c for c in XH.columns if c not in ("liv_air_n", "porta_air_n", "porta_metal_n", "phead_metal_n")]
T["p_hu_noart"] = oof_lr(XH[noart]); rec("6_group_oof", "手工特徵(去掉氣/金屬)", y.sum(), neg.sum(), auc(y, T.p_hu_noart))
demo = pd.DataFrame({"age": T.Age.fillna(T.Age.median()), "male": (T.Gender == "Male").astype(float)})
T["p_demo"] = oof_lr(demo); rec("6_group_oof", "年齡+性別", y.sum(), neg.sum(), auc(y, T.p_demo))
for base in ["E0", "E3_late"]:
    T[f"p_stack_{base}"] = oof_lr(pd.concat([XH, T[[base]]], axis=1))
    rec("6_stack_oof", f"{base} + 手工特徵", y.sum(), neg.sum(), auc(y, T[f"p_stack_{base}"]), f"base={auc(y, T[base]):.4f}")
    T[f"p_stackna_{base}"] = oof_lr(pd.concat([XH[noart], T[[base]]], axis=1))
    rec("6_stack_oof", f"{base} + 手工特徵(去氣/金屬)", y.sum(), neg.sum(), auc(y, T[f"p_stackna_{base}"]))
    T[f"p_sd_{base}"] = oof_lr(pd.concat([XH, T[[base]], demo], axis=1))
    rec("6_stack_oof", f"{base} + 手工特徵 + 年齡性別", y.sum(), neg.sum(), auc(y, T[f"p_sd_{base}"]))
wcols = [c for c in ["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"] if c in T]
T["p_stack_windows"] = oof_lr(T[wcols]); rec("5_windows", "stack 5 single-window OOF logits (LR)", y.sum(), neg.sum(), auc(y, T.p_stack_windows))
extra = [c for c in ["E3late_cat_+gb_duod", "E0_cat_all12", "E3late_cat_liver_panc"] if c in T]
if extra:
    best = max(extra + ["E3_late"], key=lambda c: auc(y, T[c]))
    T["p_stack_best"] = oof_lr(pd.concat([XH, T[[best]]], axis=1))
    rec("6_stack_oof", f"{best} + 手工特徵", y.sum(), neg.sum(), auc(y, T.p_stack_best), f"base={auc(y, T[best]):.4f}")
# stacked AUROC by subgroup (does handcrafted help mild / extrahepatic?)
for nm, pmk in [("site=extrahepatic", T.bd_site == "extrahepatic"), ("site=intrahepatic", T.bd_site == "intrahepatic"),
                ("severity=mild", T.sev == "mild"), ("moderate/marked", T.sev == "moderate/marked")]:
    mm = (pmk.values & (y == 1)) | neg
    rec("6_stack_by_sub", nm, (pmk.values & (y == 1)).sum(), neg.sum(), auc(y[mm], T.p_stack_E3_late[mm]), f"E3_late={auc(y[mm], S[mm]):.3f}")
# ---------- 7. label noise
pn = (T.bd_cat == "probably_no").values
rec("7_noise", "neg labelled probably_no (報告有描述管徑但判非擴張)", 0, pn.sum(), pn.sum() / neg.sum())
rec("7_noise", "neg not_mentioned", 0, (T.bd_cat == "not_mentioned").sum(), (T.bd_cat == "not_mentioned").mean())
rec("7_noise", "AUROC pos vs probably_no only", y.sum(), pn.sum(), auc(y[(y == 1) | pn], S[(y == 1) | pn]))
rec("7_noise", "AUROC pos vs 'no' only", y.sum(), (T.bd_cat == "no").sum(), auc(y[(y == 1) | (T.bd_cat == "no").values], S[(y == 1) | (T.bd_cat == "no").values]))
rec("7_noise", "AUROC pos vs not_mentioned only", y.sum(), (T.bd_cat == "not_mentioned").sum(),
    auc(y[(y == 1) | (T.bd_cat == "not_mentioned").values], S[(y == 1) | (T.bd_cat == "not_mentioned").values]))
neg_mention = neg & (T.rep_dilat_any == 1).values & ~pn
rec("7_noise", "neg (category no) whose report has a duct+dilat phrase (多為否定句；可疑漏標)", 0, neg_mention.sum(), neg_mention.sum() / neg.sum())
susp_pos = (y == 1) & ((T.bd_cat == "probably_yes") | ((T.qkw_prominent == 1) & (T.qkw_marked == 0)) | (T.kw_upper_normal == 1)
                       | ((T.qkw_uncertain == 1) & (T.sev != "moderate/marked"))).values
susp_neg = pn
rec("7_noise", "suspicious positives (probably_yes / prominent-only / upper-normal / uncertain-mild)", susp_pos.sum(), 0, susp_pos.sum() / y.sum())
rec("7_noise", "suspicious negatives (probably_no)", 0, susp_neg.sum(), susp_neg.sum() / neg.sum())
mm = ~(susp_pos | susp_neg)
rec("8_ceiling", "E3_late AUROC after dropping suspicious labels", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], S[mm]))
rec("8_ceiling", "stack(E3_late+手工) AUROC after dropping suspicious", y[mm].sum(), (1 - y[mm]).sum(), auc(y[mm], T.p_stack_E3_late[mm]))
clear = mm & ((y == 0) | ((T.sev == "moderate/marked") | (T.cbd_mm >= 12)).values)
rec("8_ceiling", "E3_late AUROC: clear pos (moderate/marked or CBD>=12mm) vs clean neg", y[clear].sum(), (1 - y[clear]).sum(), auc(y[clear], S[clear]))
rec("8_ceiling", "stack AUROC: clear pos vs clean neg", y[clear].sum(), (1 - y[clear]).sum(), auc(y[clear], T.p_stack_E3_late[clear]))
# ---------- 9. errors at 90% specificity (E3_late OOF)
thr = np.quantile(S[neg], 0.90); pred = S >= thr
fn = (y == 1) & ~pred; fp = neg & pred; tp = (y == 1) & pred
rec("9_err", "sensitivity@90%spec E3_late", y.sum(), neg.sum(), tp.sum() / y.sum())
thr2 = np.quantile(T.p_stack_E3_late[neg], 0.9)
rec("9_err", "sensitivity@90%spec stack(E3_late+手工)", y.sum(), neg.sum(), (T.p_stack_E3_late[y == 1] >= thr2).mean())
def cat_fn(r):
    if r.bd_cat == "probably_yes" or (r.qkw_prominent and not r.qkw_marked) or r.kw_upper_normal: return "A 標註可疑(probably_yes/僅 prominent/上限正常)"
    if r.kw_stent or r.kw_pneumobilia: return "B 已放支架/引流或氣膽"
    if r.y_chole == 1 and r.sev != "moderate/marked": return "C 膽囊切除術後輕度擴張"
    if r.bd_site == "extrahepatic": return "D 僅肝外/總膽管擴張"
    if r.sev == "mild": return "E 輕度(肝內或兩者)"
    if r.phase_hu.startswith("平掃"): return "F 平掃"
    return "G 中重度仍漏"
def cat_fp(r):
    if r.bd_cat == "probably_no": return "a 報告描述管徑但判為非擴張(probably_no)"
    if r.kw_stent or r.kw_pneumobilia: return "b 支架/氣膽"
    if r.kw_panc_mass or r.kw_pd_dilat: return "c 胰臟腫塊/胰管擴張"
    if r.y_chole == 1: return "d 膽囊切除術後"
    if r.kw_mild or r.kw_prominent: return "e 報告膽管句有 mild/prominent 字樣"
    if r.Age >= 75: return "f 高齡(>=75)"
    return "g 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
allfn = T[y == 1].apply(cat_fn, axis=1); allfp = T[neg].apply(cat_fp, axis=1)
for k, v in T.fn_cat[fn].value_counts().items():
    rec("9_err_FN", k, v, 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(allfn == k).mean():.3f}; miss rate in cat={v / (allfn == k).sum():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    rec("9_err_FP", k, 0, v, v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(allfp == k).mean():.3f}; FP rate in cat={v / (allfp == k).sum():.3f}")
# per-category AUROC (pos category vs all neg) to estimate AUROC cost
for k in sorted(allfn.unique()):
    pmk = np.zeros(len(T), bool); pmk[np.where(y == 1)[0][(allfn == k).values]] = True
    mm = pmk | neg
    rec("10_cost", f"pos {k} vs all neg", pmk.sum(), neg.sum(), auc(y[mm], S[mm]), f"frac_pos={pmk.mean() / y.mean():.3f}")
for k in sorted(allfp.unique()):
    nmk = np.zeros(len(T), bool); nmk[np.where(neg)[0][(allfp == k).values]] = True
    mm = nmk | (y == 1)
    rec("10_cost", f"all pos vs neg {k}", y.sum(), nmk.sum(), auc(y[mm], S[mm]), f"frac_neg={nmk.sum() / neg.sum():.3f}")
# AUROC if a category were removed (cost estimate = AUROC_without - AUROC_all)
base = auc(y, S)
for k in sorted(allfn.unique()):
    rm = np.zeros(len(T), bool); rm[np.where(y == 1)[0][(allfn == k).values]] = True
    rec("10_cost_drop", f"drop pos {k}", (y[~rm]).sum(), neg.sum(), auc(y[~rm], S[~rm]) - base)
for k in sorted(allfp.unique()):
    rm = np.zeros(len(T), bool); rm[np.where(neg)[0][(allfp == k).values]] = True
    rec("10_cost_drop", f"drop neg {k}", y.sum(), (neg & ~rm).sum(), auc(y[~rm], S[~rm]) - base)
res = pd.DataFrame(out); res.to_csv(A + "results_train_oof.csv", index=False)
T[["study_id", "fold", "y_bdd", "bd_cat", "bd_site", "cbd_mm", "sev", "y_chole", "phase_hu", "E0", "E3_late", "p_hu", "p_stack_E3_late",
   "fn_cat", "fp_cat"]].to_csv(A + "per_case_train_oof.csv", index=False)
# ---------- VAL descriptive only: expB probabilities by subgroup (no tuning)
V = df[df.split == "val"].copy(); V = V[V.y_bdd.notna()]
for nm in ["E0", "E3_late"]:
    z = np.load(R + f"results/expB/{nm}.npz", allow_pickle=True); fi = list(z["findings"]).index("biliary_ductal_dilation")
    spl = pd.Series(z["split"], index=z["ids"]); pv = pd.Series(z["prob"].mean(0)[:, fi], index=z["ids"])[spl == "val"]
    V[nm] = V.study_id.map(pv)
V = V[V.E3_late.notna()]; yv = V.y_bdd.values.astype(int)
V["sev"] = np.where(V.kw_marked == 1, "moderate/marked", np.where(V.kw_mild == 1, "mild", "未註明"))
vrows = []
for nm in ["E0", "E3_late"]:
    s = V[nm].values; nv = yv == 0
    def add(sub, pm=None, nm_=None):
        pm = np.ones(len(V), bool) if pm is None else np.asarray(pm, bool)
        nm_ = nv if nm_ is None else np.asarray(nm_, bool)
        mm = (pm & (yv == 1)) | (nm_ & nv)
        vrows.append(dict(model=nm, subset=sub, auc=round(auc(yv[mm], s[mm]), 4), n_pos=int((pm & (yv == 1)).sum()), n_neg=int((nm_ & nv).sum())))
    add("all")
    for v in ["intrahepatic", "both", "extrahepatic"]: add(f"pos site={v}", V.bd_site == v)
    for v in ["mild", "moderate/marked"]: add(f"pos severity={v}", V.sev == v)
    add("within 膽囊切除=1", V.y_chole == 1, V.y_chole == 1); add("within 膽囊切除=0", V.y_chole != 1, V.y_chole != 1)
    add("neg excl. probably_no", None, V.bd_cat != "probably_no")
pd.DataFrame(vrows).to_csv(A + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 400)
print(res.to_string()); print(pd.DataFrame(vrows).to_string())
