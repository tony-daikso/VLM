"""Hydronephrosis error analysis on TRAIN 5-fold OOF (val descriptive only, no tuning). Usage: hn_analyze.py <dir>
Inputs in <dir>: labels_trval.csv, hu_a6.csv, hu_ct.csv, meta_trval.csv, oof_train.csv, scores_val_descriptive.csv,
expB_val_probs.csv, gates.json, kidney_emb_trval.npz (for embedding->fluid decode)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
D = sys.argv[1].rstrip("/") + "/"
lab = pd.read_csv(D + "labels_trval.csv")
hu = pd.concat([pd.read_csv(D + "hu_a6.csv"), pd.read_csv(D + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv(D + "meta_trval.csv")
df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")

# ---------------- derived image features ----------------
def derive(d):
    for s in ("l", "r"):
        d[f"{s}_present"] = (d[f"{s}_vox"].fillna(0) >= 100).astype(int)
        for c in ("flcc", "sinus_fl", "fl_in", "shell_fl", "flcc_out", "n300_in", "sinus_n300", "shell_n300", "shell_n1000", "n1000_in", "cyst_vox"):
            d[f"{s}_{c}"] = d[f"{s}_{c}"].fillna(0)
        for c in ("flcc_ap_mm", "flcc_tr_mm", "flcc_z_mm"):
            d[f"{s}_{c}"] = d[f"{s}_{c}"].fillna(0)
        d[f"{s}_fat_in_frac"] = np.nan  # placeholder (not computed per voxel); see p10
    d["flcc_max"] = d[["l_flcc", "r_flcc"]].max(1); d["flcc_sum"] = d.l_flcc + d.r_flcc
    d["flcc_asym"] = (d.l_flcc - d.r_flcc).abs()
    d["sinus_fl_max"] = d[["l_sinus_fl", "r_sinus_fl"]].max(1)
    d["flin_frac_max"] = d[["l_fl_in_frac", "r_fl_in_frac"]].max(1)
    d["ap_max"] = d[["l_flcc_ap_mm", "r_flcc_ap_mm"]].max(1); d["tr_max"] = d[["l_flcc_tr_mm", "r_flcc_tr_mm"]].max(1)
    d["zext_max"] = d[["l_flcc_z_mm", "r_flcc_z_mm"]].max(1)
    d["flcc_out_max"] = d[["l_flcc_out", "r_flcc_out"]].max(1)
    d["p50_asym"] = (d.l_p50 - d.r_p50).abs(); d["p75_asym"] = (d.l_p75 - d.r_p75).abs()
    d["p10_min"] = d[["l_p10", "r_p10"]].min(1)
    d["vox_ratio"] = d[["l_vox", "r_vox"]].min(1) / d[["l_vox", "r_vox"]].max(1).clip(lower=1)
    d["vox_max"] = d[["l_vox", "r_vox"]].max(1)
    d["dense_coll"] = d[["l_sinus_n300", "r_sinus_n300"]].max(1) + d[["l_n300_in", "r_n300_in"]].max(1)
    d["shell_n1000_sum"] = d.l_shell_n1000 + d.r_shell_n1000 + d.l_n1000_in + d.r_n1000_in
    d["cyst_sum"] = d.l_cyst_vox + d.r_cyst_vox
    d["n_present"] = d.l_present + d.r_present
    d["kid_below_L5"] = (((d.l_z_mm - d.L5_z_mm) * np.sign(d.L3_z_mm - d.L5_z_mm) < 0) | ((d.r_z_mm - d.L5_z_mm) * np.sign(d.L3_z_mm - d.L5_z_mm) < 0)).astype(int)
    d["excretory"] = ((d.bl_n200.fillna(0) >= 100) | (d.dense_coll >= 30)).astype(int)
    ph = np.where(d.contrast == False, "非顯影", np.where(d.aorta_hu_med < 70, "非顯影(HU)", np.where(d.excretory == 1, "排泄期", np.where(d.aorta_hu_med > 250, "動脈期", "門脈期"))))
    d["phase2"] = ph
    return d
df = derive(df)
oof = pd.read_csv(D + "oof_train.csv")
T = df[df.split == "train"].merge(oof, on="study_id"); T = T[T.y_hn.notna()].reset_index(drop=True)
y = T.y_hn.values.astype(int); pos, neg = y == 1, y == 0
MODES = [c for c in ["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "E3_late", "E0_max", "E3_late_max"] if c in T]
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
out = []
def rec(sec, item, n1, n0, val, note=""):
    out.append(dict(section=sec, item=item, n_pos=int(n1), n_neg=int(n0), value=round(float(val), 4) if val == val else np.nan, note=note))
def sub(sec, name, pm, negm=neg, models=("E0", "E3_late", "E3_late_max")):
    mm = (pm & pos) | negm
    vals = {m: auc(y[mm], T[m].values[mm]) for m in models if m in T}
    rec(sec, name, (pm & pos).sum(), negm.sum(), vals.get("E3_late", np.nan),
        "; ".join(f"{k}={v:.3f}" for k, v in vals.items()) + f"; frac_of_pos={(pm & pos).sum() / pos.sum():.3f}")
    return vals
# 1 overall + gates
for m in MODES:
    rec("1_overall_oof", m, pos.sum(), neg.sum(), auc(y, T[m]))
try:
    g = json.load(open(D + "gates.json"))
    for k, v in g.items():
        rec("1_gate", k + " softmax gate (baseline,soft,liver,narrow,bone)", 0, 0, np.nan, json.dumps(np.round(np.mean(v, 0), 3).tolist()))
except Exception:
    pass
# per-window x subgroup (hyp 1)
S = T.E3_late.values
grade = T.hn_grade.values
for gname, pm in [("全部", pos), ("mild", grade == "mild"), ("moderate", grade == "moderate"), ("severe", grade == "severe"), ("未註明程度", (grade == "not_stated") & pos)]:
    vals = sub("2_grade", gname, pm, models=MODES)
side = T.hn_side.values
for sname in ["left", "right", "bilateral", "transplant", "not_stated"]:
    sub("2_side", sname, side == sname, models=MODES)
for ph in sorted(T.phase2.unique()):
    mm = (T.phase2 == ph).values
    if pos[mm].sum() >= 10:
        r = {m: auc(y[mm], T[m].values[mm]) for m in MODES}
        rec("2_phase_within", ph, pos[mm].sum(), neg[mm].sum(), r["E3_late"], "; ".join(f"{k}={v:.3f}" for k, v in r.items()))
# 3 laterality: affected vs contralateral side score (unilateral positives) + mean-pool dilution
uni = pos & np.isin(side, ["left", "right"])
for m in ["E0", "E3_late", "E3_late_max"]:
    if m + "_L" not in T: continue
    aff = np.where(side == "left", T[m + "_L"], T[m + "_R"]); con = np.where(side == "left", T[m + "_R"], T[m + "_L"])
    ok = uni & ~np.isnan(aff) & ~np.isnan(con)
    rec("3_laterality", f"{m}: P(affected-side score > contralateral) unilateral pos", ok.sum(), 0, (aff[ok] > con[ok]).mean())
for nm, pm in [("unilateral", uni), ("bilateral", pos & (side == "bilateral"))]:
    sub("3_laterality", f"{nm} pos vs all neg", pm, models=("E0", "E3_late", "E0_max", "E3_late_max"))
# image-side agreement: is the larger fluid CC on the labelled side?
ok = uni & (T.flcc_max > 0)
big = np.where(T.l_flcc > T.r_flcc, "left", "right")
rec("3_laterality", "HU: larger fluid CC is on labelled side (unilateral pos)", ok.sum(), 0, (big[ok] == side[ok]).mean())
# 4 mimics / confounders in negatives (FP rate at 90% spec of all negatives) and pos-vs-mimic AUROC
thr = np.quantile(S[neg], 0.90)
mim = {"報告提到腎外型腎盂": T.kw_extrarenal == 1, "腎盂旁囊腫": T.kw_parapelvic == 1, "腎囊腫（報告）": T.kw_renal_cyst == 1,
       "TS kidney_cyst ≥30 vox": T.cyst_sum >= 30, "輸尿管支架": T.kw_stent == 1, "腎造廔": T.kw_nephrostomy == 1,
       "移植腎": (T.kw_transplant == 1) | (T.hn_side == "transplant"), "膀胱脹大（報告）": T.kw_bladder_distended == 1,
       "排泄期（影像）": T.excretory == 1, "高密度>1000HU 於腎/周圍（支架/結石）": T.shell_n1000_sum >= 3,
       "結石/阻塞字眼": (T.kw_stone == 1) | (T.kw_obstruct == 1), "腎萎縮": T.kw_atrophy == 1, "腎切除/單腎": (T.kw_nephrectomy == 1) | (T.n_present < 2),
       "腫瘤字眼": T.kw_mass == 1, "腎盂腎炎/尿路上皮增厚": T.kw_pyelo == 1, "hn_cat=probably_no": T.hn_cat == "probably_no",
       "hn_cat=not_mentioned": T.hn_cat == "not_mentioned", "腎位於 L5 以下": T.kid_below_L5 == 1}
for k, v in mim.items():
    v = v.values
    nm_ = neg & v
    rec("4_mimic_neg", k, (pos & v).sum(), nm_.sum(), (S[nm_] >= thr).mean() if nm_.sum() else np.nan,
        f"neg rate={nm_.sum()/neg.sum():.3f}; pos rate={(pos & v).sum()/pos.sum():.3f}; AUROC pos-all vs these neg={auc(y[pos | nm_], S[pos | nm_]):.3f}; "
        f"AUROC pos-with vs all neg={auc(y[(pos & v) | neg], S[(pos & v) | neg]):.3f}; AUROC pos-without vs all neg={auc(y[(pos & ~v) | neg], S[(pos & ~v) | neg]):.3f}")
# stent shortcut: within stent studies
st = (T.kw_stent == 1).values
rec("4_stent", "AUROC within report-stent studies", (pos & st).sum(), (neg & st).sum(), auc(y[st], S[st]))
rec("4_stent", "AUROC within no-stent studies", (pos & ~st).sum(), (neg & ~st).sum(), auc(y[~st], S[~st]))
rec("4_stent", "stent flag alone AUROC", pos.sum(), neg.sum(), auc(y, st.astype(float)))
rec("4_stent", "dense>1000HU near kidney alone AUROC", pos.sum(), neg.sum(), auc(y, T.shell_n1000_sum.values))
# 5 mask coverage
for nm, mk in [("pos", pos), ("neg", neg)]:
    rec("5_mask", f"kidney p10 HU median ({nm}) [<-30 → sinus fat inside mask]", mk.sum() if nm == "pos" else 0, mk.sum() if nm == "neg" else 0, np.nanmedian(T.p10_min[mk]))
    rec("5_mask", f"frac with p10<-30 HU ({nm})", 0, 0, (T.p10_min[mk] < -30).mean())
    fc = (T.flcc_max >= 100) & mk
    insid = 1 - T.flcc_out_max[fc] / T.flcc_max[fc].clip(lower=1)
    rec("5_mask", f"largest fluid CC: median frac inside TS kidney mask ({nm}, CC≥100 vox)", fc.sum(), 0, np.median(insid) if fc.sum() else np.nan)
    rec("5_mask", f"any kidney missing (<100 vox) ({nm})", 0, 0, (T.n_present[mk] < 2).mean())
    rec("5_mask", f"kidney below L5 ({nm})", 0, 0, T.kid_below_L5[mk].mean())
    rec("5_mask", f"kidney vox median ({nm})", 0, 0, np.median(T.vox_max[mk].fillna(0)))
sub("5_mask", "both kidneys present", T.n_present.values == 2)
sub("5_mask", "a kidney missing", T.n_present.values < 2)
sub("5_mask", "transplant side label", T.hn_side.values == "transplant")
# 6 handcrafted features: single + LR / GBM OOF + stacks
HUF = ["flcc_max", "flcc_sum", "flcc_asym", "sinus_fl_max", "flin_frac_max", "ap_max", "tr_max", "zext_max", "flcc_out_max",
       "p50_asym", "p75_asym", "vox_ratio", "vox_max", "dense_coll", "shell_n1000_sum", "cyst_sum", "n_present", "kid_below_L5",
       "excretory", "aorta_hu_med", "bl_vox", "bl_p50", "bl_n200", "l_p50", "r_p50", "l_fl_in_frac", "r_fl_in_frac"]
LOGC = {"flcc_max", "flcc_sum", "flcc_asym", "sinus_fl_max", "flcc_out_max", "vox_max", "dense_coll", "shell_n1000_sum", "cyst_sum", "bl_vox", "bl_n200"}
def X_of(d, ref):
    x = d[HUF].astype(float).copy()
    for c in LOGC: x[c] = np.log1p(x[c].clip(lower=0))
    return x.fillna(ref[HUF].astype(float).median())
XH = X_of(T, T)
for f in HUF + ["Age"]:
    rec("6_single_feature", f, pos.sum(), neg.sum(), auc(y, T[f].fillna(T[f].median())))
def oof_model(Xd, kind="lr"):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = (T.fold != k).values, (T.fold == k).values
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced")) if kind == "lr" else \
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0)
        mdl.fit(Xd[tr], y[tr])
        p[te] = mdl.decision_function(Xd[te]) if kind == "lr" else mdl.predict_proba(Xd[te])[:, 1]
    return p
demo = pd.DataFrame({"age": T.Age.fillna(T.Age.median()), "female": (T.Gender == "Female").astype(float)})
T["p_hu_lr"] = oof_model(XH); T["p_hu_gbm"] = oof_model(XH, "gbm")
T["p_age"] = oof_model(demo[["age"]]); T["p_demo"] = oof_model(demo)
allw = T[[m for m in ["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"] if m in T]]
T["p_stack_windows"] = oof_model(allw)
T["p_stack_e0_hu"] = oof_model(pd.concat([XH, T[["E0"]]], axis=1))
T["p_stack_e3_hu"] = oof_model(pd.concat([XH, T[["E3_late"]]], axis=1))
T["p_stack_e3_hu_gbm"] = oof_model(pd.concat([XH, T[["E3_late"]]], axis=1), "gbm")
T["p_stack_e3max_hu"] = oof_model(pd.concat([XH, T[["E3_late_max"]]], axis=1)) if "E3_late_max" in T else np.nan
T["p_stack_e3_e3max_hu"] = oof_model(pd.concat([XH, T[["E3_late", "E3_late_max"]]], axis=1)) if "E3_late_max" in T else np.nan
T["p_stack_e3_e3max"] = oof_model(T[["E3_late", "E3_late_max"]]) if "E3_late_max" in T else np.nan
T["p_e3_demo"] = oof_model(pd.concat([T[["E3_late"]], demo], axis=1))
STK = ["p_hu_lr", "p_hu_gbm", "p_age", "p_demo", "p_stack_windows", "p_stack_e0_hu", "p_stack_e3_hu", "p_stack_e3_hu_gbm",
       "p_stack_e3max_hu", "p_stack_e3_e3max_hu", "p_stack_e3_e3max", "p_e3_demo"]
for c in STK:
    rec("7_stack_oof", c, pos.sum(), neg.sum(), auc(y, T[c]))
# grade-wise for HU and best stack
for gname, pm in [("mild", grade == "mild"), ("moderate", grade == "moderate"), ("severe", grade == "severe")]:
    mm = (pm & pos) | neg
    rec("7_stack_by_grade", gname, (pm & pos).sum(), neg.sum(), auc(y[mm], T.p_stack_e3_hu[mm]),
        f"HU-LR={auc(y[mm], T.p_hu_lr[mm]):.3f}; flcc_max={auc(y[mm], T.flcc_max[mm]):.3f}; E3={auc(y[mm], S[mm]):.3f}")
# 8 age / matched
def matched(s, reps=20, k=5):
    v = []; Ag = T.Age.values; G = T.Gender.values; ni = np.where(neg & ~np.isnan(Ag))[0]
    for r in range(reps):
        rr = np.random.default_rng(r); sel = []
        for i in np.where(pos & ~np.isnan(Ag))[0]:
            c = ni[(np.abs(Ag[ni] - Ag[i]) <= 2) & (G[ni] == G[i])]
            if len(c): sel += list(rr.choice(c, min(k, len(c)), replace=False)) + [i]
        sel = np.array(sel); v.append(auc(y[sel], np.asarray(s, float)[sel]))
    return np.mean(v)
for c in ["E0", "E3_late", "E3_late_max", "p_hu_lr", "p_stack_e3_hu", "p_stack_e3_e3max_hu", "p_age"]:
    if c in T and T[c].notna().all():
        rec("8_age", f"{c}: AUROC / age+sex matched", pos.sum(), neg.sum(), auc(y, T[c]), f"matched={matched(T[c].values):.4f}")
rec("8_age", "age median pos / neg", 0, 0, np.nanmedian(T.Age[pos]), f"neg={np.nanmedian(T.Age[neg])}")
T["agebin"] = pd.cut(T.Age, [0, 40, 60, 75, 120], labels=["<40", "40-60", "60-75", ">=75"])
for b_ in ["<40", "40-60", "60-75", ">=75"]:
    mm = (T.agebin == b_).values
    rec("8_age_strata", b_, pos[mm].sum(), neg[mm].sum(), auc(y[mm], S[mm]), f"E0={auc(y[mm], T.E0[mm]):.3f}")
for gv in ["Male", "Female"]:
    mm = (T.Gender == gv).values
    rec("8_sex_strata", gv, pos[mm].sum(), neg[mm].sum(), auc(y[mm], S[mm]))
# 9 label noise
for nm, pm in [("yes", T.hn_cat == "yes"), ("probably_yes", T.hn_cat == "probably_yes"), ("引文含 fullness/prominence", T.q_fullness == 1),
               ("引文含 pelviectasis", T.q_pelviectasis == 1), ("引文含 caliectasis", T.q_caliectasis == 1), ("引文含 hydroureter", T.q_hydroureter == 1),
               ("引文 mild/minimal", T.q_mild == 1), ("引文不確定字眼", T.q_uncertain == 1), ("引文提到 transplant", T.q_transplant == 1)]:
    sub("9_label_pos", nm, pm.values)
for nm, nm_mask in [("neg=no", T.hn_cat == "no"), ("neg=probably_no", T.hn_cat == "probably_no"), ("neg=not_mentioned", T.hn_cat == "not_mentioned")]:
    nmk = neg & nm_mask.values
    rec("9_label_neg", nm, pos.sum(), nmk.sum(), auc(y[pos | nmk], S[pos | nmk]), f"FP rate @90spec={(S[nmk] >= thr).mean():.3f}")
# suspicious negatives: big fluid CC (>= median of moderate/severe positives) & not cyst-explained
ref = np.median(T.flcc_max[pos & np.isin(grade, ["moderate", "severe"])])
susp_neg = neg & (T.flcc_max.values >= ref) & (T.cyst_sum.values < 30) & (T.kw_renal_cyst.values == 0) & (T.kw_parapelvic.values == 0)
rec("9_label_neg", f"neg with flcc_max ≥ {ref:.0f} vox (mod/sev pos median), no cyst mention", 0, susp_neg.sum(), susp_neg.mean(),
    f"FP rate@90spec={(S[susp_neg] >= thr).mean():.3f}; kw_hydroureter?={T.kw_hydroureter[susp_neg].mean():.3f}")
susp_pos = pos & ((T.hn_cat == "probably_yes") | (T.q_uncertain == 1) | (T.q_fullness == 1) | (T.q_extrarenal == 1)).values
rec("9_ceiling", "susp pos frac (probably_yes/uncertain/fullness/extrarenal)", susp_pos.sum(), 0, susp_pos.mean() / pos.mean())
mm = ~(susp_pos | susp_neg)
rec("9_ceiling", "E3_late AUROC w/o suspicious labels", pos[mm].sum(), neg[mm].sum(), auc(y[mm], S[mm]), f"stack={auc(y[mm], T.p_stack_e3_hu[mm]):.3f}")
mm2 = ((pos & np.isin(grade, ["moderate", "severe"])) | neg)
rec("9_ceiling", "E3_late AUROC moderate+severe vs all neg", pos[mm2].sum(), neg.sum(), auc(y[mm2], S[mm2]), f"stack={auc(y[mm2], T.p_stack_e3_hu[mm2]):.3f}")
# 10 errors at 90% spec
pred = S >= thr; fn = pos & ~pred; fp = neg & pred
rec("10_err", "sens@90spec E3_late", pos.sum(), neg.sum(), pred[pos].mean())
for c in ["E0", "E3_late_max", "p_stack_e3_hu", "p_stack_e3_e3max_hu"]:
    if c in T: rec("10_err", f"sens@90spec {c}", pos.sum(), neg.sum(), (T[c][pos] >= np.quantile(T[c][neg], .9)).mean())
def cat_fn(r):
    if r.n_present < 2 or r.hn_side == "transplant" or r.q_transplant: return "A 腎遮罩缺失／移植腎（不在腎遮罩內）"
    if r.hn_cat == "probably_yes" or r.q_uncertain or r.q_fullness or r.q_extrarenal: return "B 標註不確定（probably／fullness／腎外型腎盂）"
    if r.excretory: return "C 排泄期（集尿系統被顯影劑填滿）"
    if r.hn_grade == "mild" and r.flcc_max < ref: return "D 輕度且影像上積液小"
    if r.flcc_max < ref: return "E 中重度但影像積液小（遮罩/閾值未抓到）"
    return "F 影像上有明顯積液但仍漏"
def cat_fp(r):
    if r.n_present < 2: return "a 腎遮罩缺失／單腎"
    if r.cyst_sum >= 30 or r.kw_renal_cyst or r.kw_parapelvic: return "b 腎囊腫／腎盂旁囊腫"
    if r.kw_extrarenal: return "c 腎外型腎盂"
    if r.kw_stent or r.kw_nephrostomy or r.shell_n1000_sum >= 3: return "d 支架／造廔（含影像高密度）"
    if r.flcc_max >= ref: return "e 影像上有明顯積液（疑似漏標）"
    if r.kw_stone or r.kw_obstruct: return "f 結石／阻塞字眼但未擴張"
    if r.kw_atrophy or r.kw_mass: return "g 腎萎縮／腫瘤"
    return "h 其他"
T["fn_cat"] = np.where(fn, T.apply(cat_fn, axis=1), ""); T["fp_cat"] = np.where(fp, T.apply(cat_fp, axis=1), "")
basefn = T[pos].apply(cat_fn, axis=1); basefp = T[neg].apply(cat_fp, axis=1)
errs = []
for k, v in T.fn_cat[fn].value_counts().items():
    errs.append(dict(type="FN", category=k, n=int(v), frac_of_err=round(v / fn.sum(), 3), base_rate=round((basefn == k).mean(), 3)))
    rec("10_err_FN", k, v, 0, v / fn.sum(), f"of FN n={fn.sum()}; base rate in all pos={(basefn == k).mean():.3f}")
for k, v in T.fp_cat[fp].value_counts().items():
    errs.append(dict(type="FP", category=k, n=int(v), frac_of_err=round(v / fp.sum(), 3), base_rate=round((basefp == k).mean(), 3)))
    rec("10_err_FP", k, 0, v, v / fp.sum(), f"of FP n={fp.sum()}; base rate in all neg={(basefp == k).mean():.3f}")
pd.DataFrame(errs).to_csv(D + "errors_at_spec90.csv", index=False)
# 11 embedding -> fluid decode per window (hyp 1): ridge R^2, per kidney rows, train only, 5-fold
try:
    z = np.load(D + "kidney_emb_trval.npz"); eid = pd.Series(range(len(z["ids"])), index=z["ids"])
    ix = eid[T.study_id].values; E = z["E"][ix].astype(np.float32); P = z["P"][ix]
    rows = []
    for w, wn in enumerate(["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]):
        for tgt in ("flcc", "sinus_fl", "fl_in_frac"):
            Xs, ys = [], []
            for k, s in ((0, "l"), (1, "r")):
                ok = (P[:, k] > 0) & (T[f"{s}_vox"].fillna(0).values >= 100)
                Xs.append(E[ok, k, w]); ys.append(np.log1p(T[f"{s}_{tgt}"].values[ok].clip(min=0)) if tgt != "fl_in_frac" else T[f"{s}_{tgt}"].values[ok])
            Xa, ya = np.concatenate(Xs), np.concatenate(ys)
            sel = np.random.default_rng(0).choice(len(Xa), min(12000, len(Xa)), replace=False); Xa, ya = Xa[sel], ya[sel]
            pr = np.zeros(len(ya))
            for a, b in KFold(5, shuffle=True, random_state=0).split(Xa):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000])).fit(Xa[a], ya[a]); pr[b] = mdl.predict(Xa[b])
            r2 = 1 - ((ya - pr) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
            rows.append(dict(window=wn, target=tgt, r2=round(r2, 4)))
            rec("11_emb_decode", f"{wn} -> {tgt}", 0, 0, r2)
    pd.DataFrame(rows).to_csv(D + "embedding_decodes_kidney_fluid.csv", index=False)
except Exception as e:
    print("decode failed", e)
# subgroup table + per-case
res = pd.DataFrame(out); res.to_csv(D + "results_train_oof.csv", index=False)
T[["study_id", "fold", "y_hn", "hn_cat", "hn_grade", "hn_side", "phase2", "Age", "n_present", "flcc_max", "ap_max", "cyst_sum", "excretory",
   "shell_n1000_sum"] + MODES + ["p_hu_lr", "p_stack_e3_hu", "p_stack_e3_e3max_hu", "fn_cat", "fp_cat"]].to_csv(D + "per_case_train_oof.csv", index=False)
# 12 VAL descriptive: my recipe + expB + stacks fit on train OOF (no tuning)
V = df[df.split == "val"].merge(pd.read_csv(D + "scores_val_descriptive.csv"), on="study_id").merge(pd.read_csv(D + "expB_val_probs.csv"), on="study_id")
V = V[V.y_hn.notna()].reset_index(drop=True); yv = V.y_hn.values.astype(int)
vrows = [dict(model=m, subset="all", auc=auc(yv, V[m]), n_pos=yv.sum()) for m in ["expB_E0", "expB_E3_late"] + MODES if m in V]
XV = X_of(V, T)
def fit_apply(cols_tr, cols_v):
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced")).fit(cols_tr, y)
    return mdl.decision_function(cols_v)
vrows.append(dict(model="HU-LR (fit on train)", subset="all", auc=auc(yv, fit_apply(XH, XV)), n_pos=yv.sum()))
vrows.append(dict(model="stack E3_late+HU (fit on train OOF)", subset="all", auc=auc(yv, fit_apply(pd.concat([XH, T[["E3_late"]]], axis=1), pd.concat([XV, V[["E3_late"]]], axis=1))), n_pos=yv.sum()))
if "E3_late_max" in V:
    vrows.append(dict(model="stack E3_late+E3_late_max+HU", subset="all", auc=auc(yv, fit_apply(pd.concat([XH, T[["E3_late", "E3_late_max"]]], axis=1), pd.concat([XV, V[["E3_late", "E3_late_max"]]], axis=1))), n_pos=yv.sum()))
gv = V.hn_grade.values
for m in ["expB_E0", "expB_E3_late"]:
    for gname in ["mild", "moderate", "severe"]:
        mm = ((gv == gname) & (yv == 1)) | (yv == 0)
        vrows.append(dict(model=m, subset=f"{gname} vs all neg", auc=auc(yv[mm], V[m].values[mm]), n_pos=int(((gv == gname) & (yv == 1)).sum())))
    for ph in sorted(V.phase2.unique()):
        mm = (V.phase2 == ph).values
        if yv[mm].sum() >= 5: vrows.append(dict(model=m, subset=f"phase {ph}", auc=auc(yv[mm], V[m].values[mm]), n_pos=int(yv[mm].sum())))
pd.DataFrame(vrows).to_csv(D + "val_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200); pd.set_option("display.max_rows", 500)
print(res.to_string()); print(pd.DataFrame(vrows).round(4).to_string())
print("ref flcc", ref, "thr", thr)
print("phase counts train:", T.phase2.value_counts().to_dict(), "pos:", T.phase2[pos].value_counts().to_dict())
