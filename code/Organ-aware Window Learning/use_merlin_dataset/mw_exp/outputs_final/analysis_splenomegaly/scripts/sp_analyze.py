"""Splenomegaly error analysis on TRAIN 5-fold OOF (val descriptive only, nothing tuned on val; test never loaded).
Usage: sp_analyze.py <workdir with data/, oof_train.csv, scores_val.csv, geom_*.csv>  -> <workdir>/out/"""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

D = sys.argv[1]; O = D + "/out/"; os.makedirs(O, exist_ok=True)
R = {}


def A(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


lab = pd.read_csv(D + "/data/labels_trval.csv")
ok = set(lab.study_id)
meta = pd.read_csv(D + "/data/metadata_all.csv").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(ok)].drop_duplicates("study_id")[["study_id", "Age", "Gender", "phase"]]
geo = pd.concat([pd.read_csv(D + "/geom_a6.csv"), pd.read_csv(D + "/geom_ct.csv")]).drop_duplicates("study_id")
geo = geo[geo.study_id.isin(ok)]
df = lab.merge(meta, on="study_id", how="left").merge(geo, on="study_id", how="left")
df = df[df.y_sp.notna()].copy()
# derived features
df["sp_ml"] = df.sp_ml.fillna(0)
df["log_ml"] = np.log1p(df.sp_ml)
df["sp_index"] = df.sp_pc1_cm * df.sp_pc2_cm * df.sp_pc3_cm
df["cranial_cut"] = df.sp_touch_hi.fillna(0).astype(int)     # z increases cranially (spleen z > L3 z)
df["caudal_cut"] = df.sp_touch_lo.fillna(0).astype(int)
df["absent"] = (df.sp_ml < 5).astype(int)
df["vert_cm2"] = df[["L1_vert_cm2", "T12_vert_cm2", "L3_vert_cm2"]].bfill(axis=1).iloc[:, 0]
df["body_cm2"] = df[["L1_body_cm2", "T12_body_cm2", "L3_body_cm2"]].bfill(axis=1).iloc[:, 0]
df["ml_per_vert"] = df.sp_ml / df.vert_cm2
df["ml_per_body"] = df.sp_ml / df.body_cm2
df["ml_per_liver"] = df.sp_ml / df.liver_ml.replace(0, np.nan)
df["male"] = (df.Gender == "Male").astype(float)
df["cirr_or_pht"] = ((df.rp_cirrhosis == 1) | (df.rp_portal_htn == 1)).astype(int)
T = df[df.split == "train"].merge(pd.read_csv(D + "/oof_train.csv"), on="study_id").reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(D + "/scores_val.csv"), on="study_id").reset_index(drop=True)
y = T.y_sp.values.astype(int); yv = V.y_sp.values.astype(int)
R["n_train"] = len(T); R["pos_train"] = int(y.sum()); R["n_val"] = len(V); R["pos_val"] = int(yv.sum())
R["geom_missing_train"] = int(T.sp_cc_cm.isna().sum())
R["oof"] = {m: A(y, T[m]) for m in ["E0", "E3_late"]}
R["val_desc"] = {m: A(yv, V[m]) for m in ["E0", "E3_late"]}
print(R)

# ---------- 1. geometry single-feature AUROC + validity ----------
G1 = ["sp_ml", "sp_cc_cm", "sp_cc_main_cm", "sp_pc1_cm", "sp_pc2_cm", "sp_pc3_cm", "sp_ap_cm", "sp_tr_cm", "sp_max_axial_cm2",
      "sp_index", "ml_per_vert", "ml_per_body", "ml_per_liver", "sp_hu_med", "liver_ml", "body_cm2", "vert_cm2", "Age", "male"]
rows = []
for c in G1:
    for nm, M in [("train", T), ("val_desc", V)]:
        rows.append({"feature": c, "set": nm, "auroc": A(M.y_sp.astype(int), M[c]), "n_nonnull": int(M[c].notna().sum())})
sf = pd.DataFrame(rows); sf.to_csv(O + "geom_single_feature_auc.csv", index=False)
print(sf[sf.set == "train"].to_string())
# mask cc-length vs radiologist-reported length
m = T.len_cm_label.notna() & T.sp_cc_cm.notna() & (T.len_cm_label > 5)
R["len_corr_n"] = int(m.sum())
R["len_corr_cc"] = float(np.corrcoef(T.len_cm_label[m], T.sp_cc_cm[m])[0, 1])
R["len_corr_pc1"] = float(np.corrcoef(T.len_cm_label[m], T.sp_pc1_cm[m])[0, 1])
R["len_bias_cc_med"] = float((T.sp_cc_cm[m] - T.len_cm_label[m]).median())
R["len_bias_pc1_med"] = float((T.sp_pc1_cm[m] - T.len_cm_label[m]).median())
# clinical rule: length > 13 cm (cc and pc1) / volume thresholds
for c, thr in [("sp_cc_cm", 13), ("sp_pc1_cm", 13), ("sp_ml", 314), ("sp_ml", 400)]:
    p = T[c] > thr
    R[f"rule_{c}_gt{thr}"] = {"sens": float(p[y == 1].mean()), "spec": float(1 - p[y == 0].mean())}
# volume quantiles by label
R["sp_ml_q_pos"] = T.loc[y == 1, "sp_ml"].quantile([.05, .1, .25, .5, .75]).round(0).to_dict()
R["sp_ml_q_neg"] = T.loc[y == 0, "sp_ml"].quantile([.5, .75, .9, .95, .99]).round(0).to_dict()

# ---------- embedding decodes volume / length? (ridge, 5-fold on train, baseline window) ----------
z = np.load(D + "/data/spleen_emb_trval.npz"); st = np.load(D + "/data/std_stats_main12.npz")
idx = {s: i for i, s in enumerate(z["ids"])}
ii = np.array([idx[s] for s in T.study_id])
dec = []
for w, wn in enumerate(["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]):
    X = (z["E"][ii, w] - st["mu"][w]) / np.maximum(st["sd"][w], 1e-6)
    for tgt in ["log_ml", "sp_cc_cm"]:
        mm = T[tgt].notna().values & (T.sp_ml.values > 5)
        pr = np.full(len(T), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=0).split(np.where(mm)[0]):
            ia, ib = np.where(mm)[0][a], np.where(mm)[0][b]
            r = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(X[ia], T[tgt].values[ia]); pr[ib] = r.predict(X[ib])
        t = T[tgt].values[mm]; p_ = pr[mm]
        dec.append({"window": wn, "target": tgt, "R2": float(1 - ((t - p_) ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                    "MAE": float(np.abs(t - p_).mean()), "auroc_of_decoded": A(y[mm], p_)})
        if wn == "baseline":
            T["dec_" + tgt] = pr
dec = pd.DataFrame(dec); dec.to_csv(O + "embedding_decodes_volume.csv", index=False); print(dec.to_string())

# ---------- stacked models (same folds as the OOF) ----------
fold = T.fold.values


def logit(s): s = np.clip(s, -30, 30); return s


GEO = ["log_ml", "sp_cc_cm", "sp_pc1_cm", "sp_pc2_cm", "sp_pc3_cm", "sp_max_axial_cm2", "sp_ap_cm", "sp_tr_cm", "cranial_cut", "caudal_cut", "absent"]
BODY = ["vert_cm2", "body_cm2", "L1_h_cm", "liver_ml", "kid_ml", "Age", "male"]
CONF = ["Age", "male"]


def cv(cols, model="gbm", base=None):
    X = T[cols].values.astype(float) if cols else np.zeros((len(T), 0))
    if base is not None:
        X = np.c_[X, T[base].values]
    out = np.zeros(len(T))
    for k in range(5):
        a, b = fold != k, fold == k
        if model == "gbm":
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
            Xa = X
        else:
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            Xa = np.nan_to_num(X, nan=np.nanmedian(X[a], 0)[None].repeat(len(X), 0)) if np.isnan(X).any() else X
        mdl.fit(Xa[a], y[a]); out[b] = mdl.predict_proba(Xa[b])[:, 1]
    return out


models = {
    "E0": T.E0.values, "E3_late": T.E3_late.values,
    "volume_only(log_ml)": T.log_ml.values, "cc_length_only": T.sp_cc_cm.values,
    "geom_LR": cv(["log_ml", "sp_cc_cm", "sp_pc1_cm", "sp_max_axial_cm2"], "lr"),
    "geom_GBM": cv(GEO), "geom+body_GBM": cv(GEO + BODY),
    "age_sex_only_LR": cv(CONF, "lr"),
    "E3_late+geom_LR": cv(["log_ml", "sp_cc_cm", "sp_pc1_cm", "sp_max_axial_cm2", "cranial_cut"], "lr", base="E3_late"),
    "E3_late+geom_GBM": cv(GEO, base="E3_late"),
    "E3_late+geom+body_GBM": cv(GEO + BODY, base="E3_late"),
    "E3_late+body_GBM": cv(BODY, base="E3_late"),
}
mc = pd.DataFrame([{"model": k, "train_oof_auroc": A(y, v)} for k, v in models.items()])
for k, v in models.items():
    T["s_" + k] = v
# val descriptive for the main stack: fit on all train, score val
def fit_val(cols, base, model="gbm"):
    X = np.c_[T[cols].values.astype(float), T[base].values]; Xv = np.c_[V[cols].values.astype(float), V[base].values]
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0) if model == "gbm" \
        else make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if model != "gbm":
        med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X); Xv = np.where(np.isnan(Xv), med, Xv)
    return mdl.fit(X, y).predict_proba(Xv)[:, 1]
V["s_E3_late+geom_GBM"] = fit_val(GEO, "E3_late")
V["s_E3_late+geom_LR"] = fit_val(["log_ml", "sp_cc_cm", "sp_pc1_cm", "sp_max_axial_cm2", "cranial_cut"], "E3_late", "lr")
V["s_E3_late+geom+body_GBM"] = fit_val(GEO + BODY, "E3_late")
mc["val_desc_auroc"] = [A(yv, V[k]) if k in ("E0", "E3_late") else (A(yv, V["s_" + k]) if "s_" + k in V else (A(yv, V.log_ml) if k.startswith("volume") else (A(yv, V.sp_cc_cm) if k.startswith("cc_") else np.nan))) for k in mc.model]
mc.to_csv(O + "model_comparison.csv", index=False); print(mc.to_string())
BEST = "E3_late+geom_GBM"

# ---------- 2. subgroups ----------
sg = []
def sub(group, name, mask, scores=("E0", "E3_late", "s_" + BEST, "log_ml")):
    mask = np.asarray(mask, bool)
    r = {"group": group, "subgroup": name, "n": int(mask.sum()), "pos": int(y[mask].sum())}
    for s in scores:
        r[s.replace("s_", "")] = A(y[mask], T[s].values[mask])
    sg.append(r)
neg = y == 0; pos = y == 1
sub("all", "全部", np.ones(len(T)))
# positive subtypes vs all negatives
L = T.len_cm_label
ptype = np.where(T.q_kw_borderline == 1, "borderline/上限", np.where(T.q_kw_mild == 1, "mild/輕度", np.where((T.q_kw_moderate == 1) | (T.q_kw_massive == 1), "中重度/massive", "未寫程度")))
T["ptype"] = np.where(pos, ptype, "")
for t in ["borderline/上限", "mild/輕度", "未寫程度", "中重度/massive"]:
    sub("陽性程度（vs 全部陰性）", t, neg | (T.ptype == t).values)
for nm, mm in [("報告長度 <13", L < 13), ("13–13.9", (L >= 13) & (L < 14)), ("14–14.9", (L >= 14) & (L < 15)), ("15–16.9", (L >= 15) & (L < 17)), ("≥17", L >= 17), ("未寫長度", L.isna())]:
    sub("陽性報告長度 cm（vs 全部陰性）", nm, neg | (pos & mm.values))
for c in ["yes", "probably_yes"]:
    sub("陽性類別", c, neg | (T.sp_cat == c).values)
for c in ["no", "probably_no", "not_mentioned"]:
    sub("陰性類別（vs 全部陽性）", c, pos | (T.sp_cat == c).values)
sub("陰性類別（vs 全部陽性）", "no 且明寫脾臟大小正常", pos | ((T.sp_cat == "no") & (T.kw_normal_size == 1)).values)
sub("陰性類別（vs 全部陽性）", "no 但沒明寫正常", pos | ((T.sp_cat == "no") & (T.kw_normal_size == 0)).values)
for nm, mm in [("頭側被 FOV 切到", T.cranial_cut == 1), ("尾側被切到", T.caudal_cut == 1), ("未被切到", (T.cranial_cut == 0) & (T.caudal_cut == 0)),
               ("脾臟幾乎不存在（<5 mL）", T.absent == 1), ("報告提到脾切除", T.kw_splenectomy == 1), ("副脾/splenule", T.kw_accessory == 1),
               ("mask 有小型分離連通區", T.sp_ncomp_small > 0), ("脾臟病灶（梗塞/囊腫等）", T.kw_lesion == 1)]:
    sub("FOV / 解剖", nm, mm.values)
for nm, mm in [("肝硬化或門脈高壓", T.cirr_or_pht == 1), ("無肝硬化/門脈高壓", T.cirr_or_pht == 0), ("腹水", T.rp_ascites == 1),
               ("血液腫瘤相關字眼", T.rp_heme == 1), ("移植", T.rp_transplant == 1), ("淋巴結腫大", T.rp_lymphadenopathy == 1)]:
    sub("共病（報告全文）", nm, mm.values)
ab = pd.cut(T.Age, [0, 40, 55, 65, 75, 200], right=False, labels=["<40", "40–54", "55–64", "65–74", "≥75"])
for g in ab.cat.categories:
    sub("年齡", g, (ab == g).values)
for g in ["Male", "Female"]:
    sub("性別", g, (T.Gender == g).values)
bq = pd.qcut(T.body_cm2, 3, labels=["小", "中", "大"])
for g in bq.cat.categories:
    sub("體型（L1 橫切面積三分位）", g, (bq == g).values)
vq = pd.qcut(T.vert_cm2, 3, labels=["小", "中", "大"])
for g in vq.cat.categories:
    sub("椎體大小三分位", g, (vq == g).values)
SG = pd.DataFrame(sg); SG.to_csv(O + "subgroup_auroc.csv", index=False); print(SG.to_string())

# volume-matched shortcut test: within volume deciles, does cirrhosis/portal HTN raise the score among negatives?
T["vol_dec"] = pd.qcut(T.sp_ml.rank(method="first"), 10, labels=False)
sc = []
for d_ in range(10):
    mm = (T.vol_dec == d_).values
    for c in [0, 1]:
        q = mm & (T.cirr_or_pht.values == c)
        sc.append({"vol_decile": d_, "cirr_or_pht": c, "n": int(q.sum()), "pos_rate": float(y[q].mean()) if q.sum() else np.nan,
                   "mean_E3_rank_pct": float((T.E3_late.rank(pct=True).values[q]).mean()) if q.sum() else np.nan,
                   "neg_mean_E3_rank_pct": float(T.E3_late.rank(pct=True).values[q & neg].mean()) if (q & neg).sum() else np.nan})
sc = pd.DataFrame(sc); sc.to_csv(O + "shortcut_cirrhosis_by_volume_decile.csv", index=False)
# residual effect: LR of E3 logit on log_ml + cirr_or_pht (+age,sex) among all train
Xs = T[["log_ml", "sp_cc_cm", "cirr_or_pht", "Age", "male"]].copy(); Xs = Xs.fillna(Xs.median())
lr = LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(Xs), (T.E3_late > np.quantile(T.E3_late[neg], 0.9)).astype(int))
R["shortcut_lr_coef_highscore"] = dict(zip(Xs.columns, np.round(lr.coef_[0], 3).tolist()))
# label-level: does cirrhosis change P(label=1 | volume)?
lr2 = LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(Xs), y)
R["label_lr_coef"] = dict(zip(Xs.columns, np.round(lr2.coef_[0], 3).tolist()))
R["auroc_E3_minus_vol_cirr"] = {"E3_in_cirr": A(y[T.cirr_or_pht == 1], T.E3_late[T.cirr_or_pht == 1]), "vol_in_cirr": A(y[T.cirr_or_pht == 1], T.log_ml[T.cirr_or_pht == 1]),
                                "E3_in_nocirr": A(y[T.cirr_or_pht == 0], T.E3_late[T.cirr_or_pht == 0]), "vol_in_nocirr": A(y[T.cirr_or_pht == 0], T.log_ml[T.cirr_or_pht == 0])}

# body-size normalisation: does dividing by size help single-feature volume?
R["bodysize"] = {"log_ml": A(y, T.log_ml), "ml_per_vert": A(y, T.ml_per_vert), "ml_per_body": A(y, T.ml_per_body),
                 "corr_logml_vert_in_neg": float(T.loc[neg, ["log_ml", "vert_cm2"]].corr().iloc[0, 1]),
                 "corr_logml_body_in_neg": float(T.loc[neg, ["log_ml", "body_cm2"]].corr().iloc[0, 1]),
                 "corr_logml_male_in_neg": float(T.loc[neg, ["log_ml", "male"]].corr().iloc[0, 1])}
# residualise log_ml on body size among negatives (fit on train negatives), z-score
Xb = T[["vert_cm2", "body_cm2", "male", "Age"]].fillna(T[["vert_cm2", "body_cm2", "male", "Age"]].median())
from sklearn.linear_model import LinearRegression
res = np.zeros(len(T))
for k in range(5):
    a, b = (fold != k) & neg, fold == k
    lm = LinearRegression().fit(Xb[a], T.log_ml[a]); res[b] = T.log_ml[b] - lm.predict(Xb[b])
T["log_ml_resid"] = res
R["bodysize"]["log_ml_resid_bodysize"] = A(y, res)

# ---------- 3/4. error categories at fixed specificity ----------
def errs(score, spec):
    thr = np.quantile(T[score][neg], spec)
    p = T[score].values > thr
    fn = pos & ~p; fp = neg & p
    return thr, fn, fp, float(p[pos].mean())
vq95 = T.loc[neg, "sp_ml"].quantile(0.95); pmed = T.loc[pos, "sp_ml"].median(); p10 = T.loc[pos, "sp_ml"].quantile(0.10)
R["neg_p95_ml"] = float(vq95); R["pos_p10_ml"] = float(p10); R["pos_median_ml"] = float(pmed)
FNC = {"報告長度 <13.5 cm 或 borderline/mild": ((L < 13.5) | (T.q_kw_borderline == 1) | (T.q_kw_mild == 1)).values,
       "mask 體積 < 陰性 p95（影像上不大）": (T.sp_ml < vq95).values,
       "mask 頭尾長 < 12 cm": (T.sp_cc_cm < 12).values,
       "頭側被 FOV 切到": (T.cranial_cut == 1).values,
       "脾臟幾乎不存在 / 脾切除": ((T.absent == 1) | (T.kw_splenectomy == 1)).values,
       "肝硬化/門脈高壓": (T.cirr_or_pht == 1).values,
       "措辭不確定或 probably_yes": ((T.q_kw_uncertain == 1) | (T.sp_cat == "probably_yes")).values,
       "報告只在病史/臨床提及": (T.kw_history == 1).values}
FPC = {"mask 體積 ≥ 陽性 p10（影像上偏大）": (T.sp_ml >= p10).values,
       "mask 頭尾長 ≥ 13 cm": (T.sp_cc_cm >= 13).values,
       "not_mentioned": (T.sp_cat == "not_mentioned").values,
       "no 且明寫大小正常": ((T.sp_cat == "no") & (T.kw_normal_size == 1)).values,
       "報告有寫長度 ≥13 卻被判陰性": (T.len_cm_report >= 13).values,
       "肝硬化/門脈高壓": (T.cirr_or_pht == 1).values,
       "報告有 mild/borderline 字眼": ((T.kw_mild == 1) | (T.kw_borderline == 1)).values,
       "體型大（L1 面積上三分位）": (bq == "大").values,
       "男性": (T.male == 1).values,
       "脾臟病灶": (T.kw_lesion == 1).values}
ec = []
for score in ["E3_late", "s_" + BEST]:
    for spec in [0.90, 0.95, 0.98]:
        thr, fn, fp, sens = errs(score, spec)
        R[f"sens_{score}_spec{spec}"] = sens
        for k, v in FNC.items():
            ec.append({"score": score.replace("s_", ""), "spec": spec, "type": "FN", "n_err": int(fn.sum()), "category": k, "n": int((fn & v).sum()),
                       "frac_of_errors": float((fn & v).sum() / max(1, fn.sum())), "frac_in_all_pos": float(v[pos].mean())})
        for k, v in FPC.items():
            ec.append({"score": score.replace("s_", ""), "spec": spec, "type": "FP", "n_err": int(fp.sum()), "category": k, "n": int((fp & v).sum()),
                       "frac_of_errors": float((fp & v).sum() / max(1, fp.sum())), "frac_in_all_neg": float(v[neg].mean())})
EC = pd.DataFrame(ec); EC.to_csv(O + "errors_at_spec.csv", index=False)
print(EC[(EC.score == "E3_late") & (EC.spec == 0.90)].to_string())

# ---------- AUROC cost per error source: drop the subgroup, recompute ----------
base = A(y, T.E3_late)
cost = []
DROP = {"陽性 borderline/mild（措辭）": pos & ((T.q_kw_borderline == 1) | (T.q_kw_mild == 1)).values,
        "陽性 報告長度 <13.5 cm": pos & (L < 13.5).values,
        "陽性 mask 體積 < 陰性 p95": pos & (T.sp_ml < vq95).values,
        "陰性 mask 體積 ≥ 陽性 p10（疑似漏標/上限）": neg & (T.sp_ml >= p10).values,
        "陰性 not_mentioned": neg & (T.sp_cat == "not_mentioned").values,
        "陰性 not_mentioned 且體積 ≥ 陽性 p10": neg & (T.sp_cat == "not_mentioned").values & (T.sp_ml >= p10).values,
        "頭側被 FOV 切到（全部）": (T.cranial_cut == 1).values,
        "脾臟缺如/脾切除（全部）": ((T.absent == 1) | (T.kw_splenectomy == 1)).values,
        "肝硬化/門脈高壓（全部）": (T.cirr_or_pht == 1).values,
        "措辭不確定 / probably_yes 陽性": pos & ((T.q_kw_uncertain == 1) | (T.sp_cat == "probably_yes")).values}
for k, v in DROP.items():
    keep = ~v
    cost.append({"drop": k, "n_dropped": int(v.sum()), "pos_dropped": int((v & pos).sum()), "E3_late_after": A(y[keep], T.E3_late[keep]),
                 "gain": A(y[keep], T.E3_late[keep]) - base, "stack_after": A(y[keep], T["s_" + BEST][keep]),
                 "volume_only_after": A(y[keep], T.log_ml[keep])})
CO = pd.DataFrame(cost).sort_values("gain", ascending=False); CO.to_csv(O + "auroc_cost_drop_subgroup.csv", index=False); print(CO.to_string())

# ---------- ceiling ----------
# clean subset: explicit positives (yes, no borderline/mild/uncertain) vs explicit 'no' with stated normal size
cl_pos = pos & (T.sp_cat == "yes").values & (T.q_kw_borderline == 0).values & (T.q_kw_uncertain == 0).values
cl_neg = neg & (T.sp_cat == "no").values
R["clean_subset"] = {"n_pos": int(cl_pos.sum()), "n_neg": int(cl_neg.sum()),
                     **{k: A(y[cl_pos | cl_neg], T[s][cl_pos | cl_neg]) for k, s in [("E3_late", "E3_late"), ("stack", "s_" + BEST), ("volume", "log_ml")]}}
# noise ceiling: fraction of negatives whose volume looks enlarged beyond the positive median (likely missed) and positives below the neg median
a_ = float(((T.sp_ml < T.loc[neg, "sp_ml"].median()) & pos).sum() / pos.sum())
b_ = float(((T.sp_ml > pmed) & neg).sum() / neg.sum())
R["noise_a_pos_below_negmedian"] = a_; R["noise_b_neg_above_posmedian"] = b_
# AUROC of a perfect detector when a frac of positives are 'really negative' and b of negatives 'really positive'
P_, N_ = pos.sum(), neg.sum()
tp_true = P_ * (1 - a_); fp_true = N_ * b_
# true-pos ranked above all; mislabeled items interleave with their true class
auc = ((tp_true * (N_ - fp_true)) + (tp_true * fp_true) * 0.5 + (P_ * a_ * fp_true) * 0 + (P_ * a_) * (N_ - fp_true) * 0.5) / (P_ * N_)
R["noise_ceiling_auroc"] = float(auc)

# ---------- per-case outputs (no text) ----------
keep_cols = ["study_id", "fold", "y_sp", "sp_cat", "len_cm_label", "sp_ml", "sp_cc_cm", "sp_pc1_cm", "cranial_cut", "absent", "cirr_or_pht",
             "E0", "E3_late", "s_" + BEST, "s_geom_GBM"]
T[keep_cols].to_csv(O + "per_case_train_oof.csv", index=False)
V[["study_id", "y_sp", "E0", "E3_late", "s_" + BEST, "s_E3_late+geom_LR"]].to_csv(O + "scores_val_descriptive.csv", index=False)
R["val_desc_subgroups"] = {"cranial_cut": A(yv[V.cranial_cut == 1], V.E3_late[V.cranial_cut == 1]), "n_val_cranial_cut": int((V.cranial_cut == 1).sum()),
                           "volume_only": A(yv, V.log_ml), "cc_len_only": A(yv, V.sp_cc_cm)}
R["counts"] = {"cranial_cut_train": int(T.cranial_cut.sum()), "cranial_cut_pos": int(T.cranial_cut[pos].sum()), "absent_train": int(T.absent.sum()),
               "splenectomy_kw": int(T.kw_splenectomy.sum()), "accessory_kw": int(T.kw_accessory.sum()), "cirr_or_pht": int(T.cirr_or_pht.sum()),
               "cirr_or_pht_pos": int(T.cirr_or_pht[pos].sum()), "pos_len_reported": int(L[pos].notna().sum()),
               "neg_notmentioned": int((T.sp_cat == "not_mentioned").sum()), "pos_borderline_or_mild": int((pos & ((T.q_kw_borderline == 1) | (T.q_kw_mild == 1)).values).sum())}
json.dump(R, open(O + "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
