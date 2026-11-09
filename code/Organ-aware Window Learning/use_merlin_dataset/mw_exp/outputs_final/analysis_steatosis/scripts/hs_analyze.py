"""Steatosis error analysis on TRAIN OOF (val descriptive only). Usage: hs_analyze.py <dir with data/, oof_train.csv, hs_*.csv>"""
import sys, json, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
D = sys.argv[1]; O = D + "/out/"
import os; os.makedirs(O, exist_ok=True)
lab = pd.read_csv(D + "/data/labels_trval.csv")
lab = lab[lab.split.isin(["train", "val"])]
ok = set(lab.study_id)
meta = pd.read_csv(D + "/data/metadata_all.csv").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(ok)].drop_duplicates("study_id")[["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "kvp", "slicethickness"]]
hu = pd.concat([pd.read_csv(D + "/hs_a6.csv"), pd.read_csv(D + "/hs_ct.csv")]).drop_duplicates("study_id")
hu = hu[hu.study_id.isin(ok)]
df = lab.merge(meta, on="study_id", how="left").merge(hu, on="study_id", how="left")
df = df[df.y_hs.notna()].copy()

# ---------------- derived features ----------------
df["LS_diff"] = df.liver_er_med - df.spleen_med
df["LS_ratio"] = df.liver_er_med / df.spleen_med.where(df.spleen_med.abs() > 5)
df["L_minus_aorta"] = df.liver_er_med - df.aorta_med
df["L_minus_portal"] = df.liver_er_med - df.portal_med
df["kid_med"] = df[["kidney_left_med", "kidney_right_med"]].mean(1)
df["kid_cmd"] = (df[["kidney_left_p90", "kidney_right_p90"]].mean(1) - df[["kidney_left_p10", "kidney_right_p10"]].mean(1))
df["log_liver_vox"] = np.log(df.liver_vox.clip(lower=1))
df["male"] = (df.Gender == "Male").astype(float)
df["fat_frac_L3"] = df.L3_fat_cm2 / df.L3_body_cm2
# metadata phase
ph = np.where(df.contrast.astype(str).str.lower().isin(["false", "0", "0.0"]), "native", df.phase.fillna("unknown"))
df["phase_meta"] = pd.Series(ph, index=df.index).replace({"native": "native"})
# HU-estimated phase
a, p = df.aorta_med, df.portal_med
est = np.select([(a < 80) & ~(p > 100), (a >= 180) & ((a - p) > 50), a < 110], ["native", "arterial", "delayed"], "portal_venous")
est = np.where(a.isna(), "unknown", est)
df["phase_hu"] = est
df["liver_seg_bad"] = ((df.liver_cut == 1) | (df.mask_agree < 0.9) | (df.liver_vox < df.liver_vox.quantile(0.02))).astype(int)

T = df[df.split == "train"].merge(pd.read_csv(D + "/oof_train.csv"), on="study_id").merge(pd.read_csv(D + "/oof_train_concat_diff.csv").drop(columns="fold"), on="study_id").reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(D + "/scores_val_descriptive.csv"), on="study_id").merge(pd.read_csv(D + "/scores_val_concat_diff.csv"), on="study_id").reset_index(drop=True)
y = T.y_hs.values.astype(int)
R = {}  # json results
rows = []  # subgroup table


def A(yy, ss):
    yy = np.asarray(yy); ss = np.asarray(ss); m = ~np.isnan(ss)
    yy, ss = yy[m], ss[m]
    return roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan


def sub(group, name, mask, cols=("E0", "E3_late")):
    m = np.asarray(mask, bool)
    r = {"group": group, "subgroup": name, "n": int(m.sum()), "n_pos": int(y[m].sum()), "prev": round(y[m].mean(), 3) if m.sum() else np.nan}
    for c in cols:
        r[c] = round(A(y[m], T[c].values[m]), 4)
    rows.append(r)


sub("overall", "all", np.ones(len(T)))
print("OOF overall:", rows[-1])

# ---------------- H1 phase ----------------
R["phase_crosstab"] = pd.crosstab(T.phase_meta, T.phase_hu).to_dict()
print(pd.crosstab(T.phase_meta, T.phase_hu))
for c in ["phase_meta", "phase_hu"]:
    for g in sorted(T[c].unique()):
        sub(c, g, T[c] == g)

# ---------------- handcrafted HU features ----------------
HU = ["liver_er_med", "liver_mean", "liver_p10", "liver_p25", "liver_p50", "liver_p75", "liver_p90", "liver_std", "liver_er_iqr",
      "liver_frac_lt40", "liver_frac_lt0", "liver_slab_sd", "spleen_med", "LS_diff", "LS_ratio", "aorta_med", "portal_med", "ivc_med",
      "kid_med", "kid_cmd", "L_minus_aorta", "L_minus_portal", "log_liver_vox", "L3_fat_cm2", "L3_fat_outer_cm2", "L3_fat_inner_cm2",
      "fat_frac_L3", "L1_fat_cm2", "Age", "male"]
single = []
for f in HU:
    s = T[f].values.astype(float)
    r = {"feature": f}
    for g in ["all", "native", "portal_venous", "arterial", "delayed"]:
        m = np.ones(len(T), bool) if g == "all" else (T.phase_hu == g).values
        au = A(y[m], s[m])
        r[f"auc_{g}"] = round(max(au, 1 - au), 4) if not np.isnan(au) else np.nan
        r[f"dir_{g}"] = "+" if (not np.isnan(au) and au >= 0.5) else "-"
    single.append(r)
single = pd.DataFrame(single).sort_values("auc_all", ascending=False)
single.to_csv(O + "hu_single_feature_auc.csv", index=False)
print(single.head(12).to_string())

PH = ["native", "portal_venous", "arterial", "delayed"]


def design(d, feats, inter=True, logit_cols=()):
    X = d[feats].astype(float).copy()
    X = X.fillna(T[feats].astype(float).median())
    for g in PH:
        X[f"ph_{g}"] = (d.phase_hu == g).astype(float) if g != "delayed" else (d.phase_hu == "delayed").astype(float)
    if inter:
        for g in PH:
            for f in [f for f in ("liver_er_med", "LS_diff", "liver_p25", "L_minus_portal") if f in feats]:
                X[f"{f}_x_{g}"] = X[f] * X[f"ph_{g}"]
    for c in logit_cols:
        X[c] = d[c].values
    return X


def oof_model(make, feats, inter=True, logit_cols=(), name=""):
    X = design(T, feats, inter, logit_cols).values
    s = np.zeros(len(T))
    for k in range(5):
        tr, te = T.fold.values != k, T.fold.values == k
        s[te] = make().fit(X[tr], y[tr]).decision_function(X[te]) if hasattr(make(), "decision_function") else make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    mdl = make().fit(X, y)
    Xv = design(V, feats, inter, logit_cols).values
    sv = mdl.decision_function(Xv) if hasattr(mdl, "decision_function") else mdl.predict_proba(Xv)[:, 1]
    return s, sv


LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced"))
GB = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
HUc = [f for f in HU if f not in ("Age", "male")]
models = {
    "HU_liver_only_LR": (LR, ["liver_er_med", "liver_p25", "liver_p50", "liver_p75", "liver_frac_lt40"], False, ()),
    "HU_LS_phase_LR": (LR, ["liver_er_med", "spleen_med", "LS_diff", "LS_ratio", "aorta_med", "portal_med", "liver_p25"], True, ()),
    "HU_all_LR": (LR, HUc, True, ()),
    "HU_all_GBM": (GB, HUc + ["Age", "male"], False, ()),
    "E0+HU_LR": (LR, HUc, True, ("E0",)),
    "E3_late+HU_LR": (LR, HUc, True, ("E3_late",)),
    "E3_late+HU_GBM": (GB, HUc + ["Age", "male"], False, ("E3_late",)),
    "E3_late_LSconcat+HU_GBM": (GB, HUc + ["Age", "male"], False, ("E3_late_concat_diff",)),
    "E3_late+E1all+HU_GBM": (GB, HUc + ["Age", "male"], False, ("E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone")),
}
mres = []
for nm, (mk, feats, inter, lc) in models.items():
    s, sv = oof_model(mk, feats, inter, lc, nm)
    T[nm] = s; V[nm] = sv
    r = {"model": nm, "train_oof_auc": round(A(y, s), 4), "val_desc_auc": round(A(V.y_hs.values, sv), 4)}
    for g in PH:
        m = (T.phase_hu == g).values
        r[f"oof_{g}"] = round(A(y[m], s[m]), 4)
    mres.append(r); print(r, flush=True)
for c in ["E0", "E3_late", "E3_late_concat_diff", "E0_concat_diff", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]:
    r = {"model": c, "train_oof_auc": round(A(y, T[c]), 4), "val_desc_auc": round(A(V.y_hs.values, V[c]), 4)}
    for g in PH:
        m = (T.phase_hu == g).values
        r[f"oof_{g}"] = round(A(y[m], T[c].values[m]), 4)
    mres.append(r)
pd.DataFrame(mres).to_csv(O + "model_comparison.csv", index=False)
print(pd.DataFrame(mres).to_string())

# ---------------- H2: can the embedding linearly decode absolute liver HU? ----------------
z = np.load(D + "/data/liver_emb_trval.npz"); st = np.load(D + "/data/std_stats_main12.npz")
eid = {s: i for i, s in enumerate(z["ids"])}
ix = np.array([eid[s] for s in T.study_id])
Eall = (z["E"][ix] - st["mu"]) / np.maximum(st["sd"], 1e-6)
tgt = T.liver_er_med.values; good = ~np.isnan(tgt)
dec = []
for wi, w in enumerate(["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]):
    X = Eall[:, wi]; pred = np.full(len(T), np.nan)
    for k in range(5):
        tr, te = (T.fold.values != k) & good, (T.fold.values == k) & good
        pred[te] = RidgeCV(alphas=[1, 10, 100, 1000]).fit(X[tr], tgt[tr]).predict(X[te])
    r2 = 1 - np.nansum((tgt - pred) ** 2) / np.nansum((tgt[good] - tgt[good].mean()) ** 2)
    pvm = good & (T.phase_hu == "portal_venous").values
    r2n = 1 - np.sum((tgt[pvm] - pred[pvm]) ** 2) / np.sum((tgt[pvm] - tgt[pvm].mean()) ** 2)
    mae = np.nanmean(np.abs(tgt - pred))
    dec.append({"window": w, "R2_liverHU_all": round(r2, 3), "R2_liverHU_PV": round(r2n, 3), "MAE_HU": round(mae, 1),
                "auc_decodedHU_all": round(1 - A(y[good], pred[good]), 4)})
    T[f"decHU_{w}"] = pred
tg2 = T.LS_diff.values; g2 = ~np.isnan(tg2)
for wi, w in enumerate(["baseline", "liver"]):
    pred = np.full(len(T), np.nan)
    for k in range(5):
        tr, te = (T.fold.values != k) & g2, (T.fold.values == k) & g2
        pred[te] = RidgeCV(alphas=[1, 10, 100, 1000]).fit(Eall[:, wi * 2 if w == "liver" else 0][tr], tg2[tr]).predict(Eall[:, 2 if w == "liver" else 0][te])
    dec.append({"window": w + " -> LS_diff", "R2_liverHU_all": round(1 - np.nansum((tg2 - pred) ** 2) / np.nansum((tg2[g2] - tg2[g2].mean()) ** 2), 3),
                "MAE_HU": round(np.nanmean(np.abs(tg2 - pred)), 1), "auc_decodedHU_all": round(1 - A(y[g2], pred[g2]), 4)})
pd.DataFrame(dec).to_csv(O + "embedding_decodes_liver_hu.csv", index=False)
print(pd.DataFrame(dec).to_string())

# ---------------- H3 subtypes (positives subgroup vs all negatives) ----------------
neg = y == 0
grade = T.hs_detail.fillna("{}").map(lambda s: json.loads(s).get("grade", "na"))
T["grade"] = grade
for g in sorted(grade[y == 1].unique()):
    sub("grade(pos) vs all neg", g, neg | ((y == 1) & (grade == g).values))
for k in ["kw_focal", "kw_hetero", "kw_diffuse", "kw_mild", "kw_severe", "kw_uncertain", "kw_decr_att", "kw_steatosis", "kw_fatty"]:
    for v in (1, 0):
        sub(f"pos {k} vs all neg", f"{k}={v}", neg | ((y == 1) & (T[k].values == v)))
st = np.select([T.pure_focal == 1, T.q_kw_sparing == 1, (T.q_kw_steatosis == 1) | (T.q_kw_diffuse == 1), (T.q_kw_fatty == 1), T.q_kw_decr_att == 1],
               ["局部脂肪浸潤(無瀰漫字眼)", "瀰漫+局部sparing", "瀰漫/steatosis", "fatty 其他措辭", "僅'低密度'措辭"], "其他")
T["subtype"] = st
for g in pd.Series(st[y == 1]).value_counts().index:
    sub("subtype(pos) vs all neg", g, neg | ((y == 1) & (st == g)))
R["subtype_counts_pos"] = pd.Series(st[y == 1]).value_counts().to_dict()
for g in ["瀰漫/steatosis", "局部脂肪浸潤(無瀰漫字眼)"]:
    for nm in ["liver_er_med", "LS_diff"]:
        R[f"median_{nm}_PV_{g}"] = float(T.loc[(y == 1) & (st == g) & (T.phase_hu == "portal_venous").values, nm].median())
for nm in ["liver_er_med", "LS_diff"]:
    R[f"median_{nm}_PV_neg"] = float(T.loc[(y == 0) & (T.phase_hu == "portal_venous").values, nm].median())
for c in ("yes", "probably_yes"):
    sub("label category(pos) vs all neg", c, neg | (T.hs_cat == c).values)
for c in ("no", "not_mentioned", "probably_no"):
    sub("all pos vs neg category", c, (y == 1) | (T.hs_cat == c).values)

# ---------------- H4 label noise ----------------
nat = (T.phase_hu == "native").values
hu_pos = ((T.liver_er_med < 40) | (T.LS_diff < -10)).values & ~T.liver_er_med.isna().values
ln = {}
ln["native_n"] = int(nat.sum())
ln["native_HUcrit_pos_rate_in_labelpos"] = float(hu_pos[nat & (y == 1)].mean())
ln["native_HUcrit_pos_rate_in_labelneg"] = float(hu_pos[nat & (y == 0)].mean())
ln["native_HUcrit_pos_rate_in_neg_not_mentioned"] = float(hu_pos[nat & (T.hs_cat == "not_mentioned").values].mean())
ln["native_HUcrit_pos_rate_in_neg_no"] = float(hu_pos[nat & (T.hs_cat == "no").values].mean())
ln["native_HUcrit_auc(binary)"] = float(A(y[nat], hu_pos[nat].astype(float)))
ln["native_liverHU_auc"] = float(1 - A(y[nat], T.liver_er_med.values[nat]))
ln["neg_with_steatosis_word"] = int(((y == 0) & (T.kw_steatosis == 1).values).sum())
ln["neg_with_fatty_word"] = int(((y == 0) & (T.kw_fatty == 1).values).sum())
ln["pos_uncertain_or_probably"] = int(((y == 1) & ((T.kw_uncertain == 1) | (T.hs_cat == "probably_yes")).values).sum())
ln["pos_steat_only_history"] = int(((y == 1) & (T.steat_only_history == 1).values).sum())
ln["pos_quote_empty"] = int(((y == 1) & (T.quote_empty == 1).values).sum())
ln["contrast_limit_any"] = int((T.kw_contrast_limit == 1).sum())
ln["neg_iron_or_hyperatt"] = int(((y == 0) & (T.kw_iron == 1).values).sum())
ln["n_pos"] = int(y.sum()); ln["n_neg"] = int((y == 0).sum())
# noise model on native: P(label pos | HU very low) etc
vlow = nat & (T.liver_er_med < 30).values & (T.LS_diff < -15).values
vhigh = nat & (T.liver_er_med > 55).values & (T.LS_diff > 0).values
ln["native_very_low_HU_n"] = int(vlow.sum()); ln["native_very_low_HU_labelpos_rate"] = float(y[vlow].mean())
ln["native_normal_HU_n"] = int(vhigh.sum()); ln["native_normal_HU_labelpos_rate"] = float(y[vhigh].mean())
pv = (T.phase_hu == "portal_venous").values
pvc = pv & (T.LS_diff < -25).values      # PV-phase steatosis criterion (L-S < -25 HU)
pvn = pv & (T.LS_diff > 5).values & (T.liver_er_med > 100).values
ln["pv_n"] = int(pv.sum())
ln["pv_LSlt-25_n"] = int(pvc.sum()); ln["pv_LSlt-25_labelpos_rate"] = float(y[pvc].mean())
ln["pv_LSgt5_liver>100_n"] = int(pvn.sum()); ln["pv_LSgt5_liver>100_labelpos_rate"] = float(y[pvn].mean())
ln["pv_LSlt-25_rate_in_labelpos"] = float(pvc[pv & (y == 1)].mean()); ln["pv_LSlt-25_rate_in_labelneg"] = float(pvc[pv & (y == 0)].mean())
ln["pv_LSlt-25_rate_in_neg_not_mentioned"] = float(pvc[pv & (T.hs_cat == "not_mentioned").values].mean())
ln["pv_LSlt-25_rate_in_neg_no"] = float(pvc[pv & (T.hs_cat == "no").values].mean())
ln["pv_LSlt-25_neg_with_E3_top10pct"] = int((pvc & (y == 0) & (T.E3_late.values >= np.quantile(T.E3_late.values[y == 0], 0.9))).sum())
for k, v in ln.items():
    print("LN", k, v)
R["label_noise"] = ln

# ---------------- H5 segmentation ----------------
for nm, m in [("liver_cut(FOV partial)", T.liver_cut == 1), ("mask_agree<0.9", T.mask_agree < 0.9),
              ("liver_vox<p2", T.liver_vox < T.liver_vox.quantile(0.02)), ("liver_vox>p98", T.liver_vox > T.liver_vox.quantile(0.98)),
              ("seg_bad_any", T.liver_seg_bad == 1), ("seg_ok", T.liver_seg_bad == 0)]:
    sub("segmentation", nm, m.values)

# ---------------- H6 windows x phase ----------------
for w in ["E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]:
    for g in ["native", "portal_venous"]:
        m = (T.phase_hu == g).values
        rows.append({"group": "window x phase", "subgroup": f"{w} | {g}", "n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "prev": round(y[m].mean(), 3), "E0": np.nan, "E3_late": round(A(y[m], T[w].values[m]), 4)})

# enhancement strata
aq = pd.qcut(T.aorta_med, 4, labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])
for g in aq.cat.categories:
    sub("aorta HU quartile", g, (aq == g).values)
lq = pd.qcut(T.portal_med, 4, labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])
for g in lq.cat.categories:
    sub("portal HU quartile", g, (lq == g).values)
# ---------------- H7 confounders ----------------
ab = pd.cut(T.Age, [0, 40, 55, 65, 75, 120], labels=["<40", "40-54", "55-64", "65-74", ">=75"])
for g in ab.cat.categories:
    sub("age", g, (ab == g).values)
for g in ("Male", "Female"):
    sub("sex", g, (T.Gender == g).values)
q = pd.qcut(T.L3_fat_cm2, 3, labels=["low", "mid", "high"])
for g in q.cat.categories:
    sub("L3 fat area tertile", g, (q == g).values)
for g in T.manufacturer.value_counts().index[:3]:
    sub("manufacturer", g, (T.manufacturer == g).values)
# confounder-only model
s_conf, _ = oof_model(LR, ["Age", "male", "L3_fat_cm2", "L3_fat_inner_cm2", "fat_frac_L3", "log_liver_vox"], False, (), "conf")
R["confounder_only_auc"] = float(A(y, s_conf))
R["fat_area_auc"] = float(A(y, T.L3_fat_cm2.fillna(T.L3_fat_cm2.median())))
# AUROC of E3 within confounder-matched pairs: partial out confounders
print("confounder-only OOF AUROC", R["confounder_only_auc"])

sg = pd.DataFrame(rows)
sg.to_csv(O + "subgroup_auroc.csv", index=False)
print(sg.to_string())

# ---------------- error categories at 90% specificity (E3_late train OOF) ----------------
best = "E3_late+HU_GBM"
err = []
for sc in ["E3_late", best]:
    s = T[sc].values; thr = np.quantile(s[y == 0], 0.90)
    fn = (y == 1) & (s < thr); fp = (y == 0) & (s >= thr)
    R[f"sens_at_spec90_{sc}"] = float(((y == 1) & (s >= thr)).sum() / (y == 1).sum())
    for kind, m in (("FN", fn), ("FP", fp)):
        d = T[m]
        cats = {}
        if kind == "FN":
            cats["造影期(非 native)"] = (d.phase_hu != "native")
            cats["native 但肝 HU≥45 且 L−S≥−5（影像不支持）"] = (d.phase_hu == "native") & (d.liver_er_med >= 45) & (d.LS_diff >= -5)
            cats["輕度 (grade=mild)"] = d.grade == "mild"
            cats["局部脂肪浸潤(無瀰漫字眼)"] = d.subtype == "局部脂肪浸潤(無瀰漫字眼)"
            cats["PV 期且 L−S≥−5（影像上肝不比脾暗）"] = (d.phase_hu == "portal_venous") & (d.LS_diff >= -5)
            cats["高顯影(portal HU 最高四分位)"] = d.portal_med > T.portal_med.quantile(0.75)
            cats["局部/不均勻/sparing"] = (d.kw_focal == 1) | (d.kw_hetero == 1)
            cats["不確定語氣或 probably_yes"] = (d.kw_uncertain == 1) | (d.hs_cat == "probably_yes")
            cats["肝分割問題"] = d.liver_seg_bad == 1
        else:
            cats["造影期(非 native)"] = (d.phase_hu != "native")
            cats["native 且 HU 標準陽性（疑似漏標）"] = (d.phase_hu == "native") & ((d.liver_er_med < 40) | (d.LS_diff < -10))
            cats["報告含 steatosis/fatty 字眼"] = (d.kw_steatosis == 1) | (d.kw_fatty == 1)
            cats["not_mentioned（未提及）"] = d.hs_cat == "not_mentioned"
            cats["肝硬化/不均勻/肝腫大"] = (d.kw_cirrhosis == 1) | (d.kw_hetero == 1) | (d.kw_hepatomegaly == 1)
            cats["肝分割問題"] = d.liver_seg_bad == 1
            cats["PV 期 L−S<−25（影像像脂肪肝，疑似漏標）"] = (d.phase_hu == "portal_venous") & (d.LS_diff < -25)
            cats["低顯影(portal HU 最低四分位)"] = d.portal_med < T.portal_med.quantile(0.25)
            cats["L3 脂肪面積最高三分位"] = d.L3_fat_cm2 > T.L3_fat_cm2.quantile(2 / 3)
        for c, mm in cats.items():
            err.append({"score": sc, "type": kind, "n_type": int(m.sum()), "category": c, "n": int(mm.sum()), "pct": round(100 * mm.mean(), 1)})
pd.DataFrame(err).to_csv(O + "errors_at_spec90.csv", index=False)
print(pd.DataFrame(err).to_string())

# ---------------- AUROC cost estimates: remove subgroup ----------------
cost = []
base = A(y, T.E3_late)
def drop(nm, m):
    keep = ~np.asarray(m, bool)
    cost.append({"factor": nm, "n_removed": int((~keep).sum()), "pos_removed": int(y[~keep].sum()),
                 "E3_late_auc_without": round(A(y[keep], T.E3_late.values[keep]), 4), "delta": round(A(y[keep], T.E3_late.values[keep]) - base, 4)})
drop("neg: steatosis/fatty word in report", (y == 0) & ((T.kw_steatosis == 1) | (T.kw_fatty == 1)).values)
drop("neg: native HU-criterion positive", (y == 0) & nat & hu_pos)
drop("pos: probably_yes/uncertain", (y == 1) & ((T.kw_uncertain == 1) | (T.hs_cat == "probably_yes")).values)
drop("pos: pure focal fat", (y == 1) & (T.subtype == "局部脂肪浸潤(無瀰漫字眼)").values)
drop("neg: PV L-S<-25 (possible missed)", (y == 0) & (T.phase_hu == "portal_venous").values & (T.LS_diff < -25).values)
drop("high enhancement (portal HU top quartile)", (T.portal_med > T.portal_med.quantile(0.75)).values)
drop("pos: mild", (y == 1) & (T.grade == "mild").values)
drop("pos: focal/hetero", (y == 1) & ((T.kw_focal == 1) | (T.kw_hetero == 1)).values)
drop("liver seg bad (all)", (T.liver_seg_bad == 1).values)
drop("non-native (keep native only)", ~nat)
drop("native only removed", nat)
drop("not_mentioned negatives", (T.hs_cat == "not_mentioned").values)
pd.DataFrame(cost).to_csv(O + "auroc_cost_drop_subgroup.csv", index=False)
print(pd.DataFrame(cost).to_string())

# ---------------- ceiling ----------------
# perfect-detector AUROC under label noise a (pos that are truly neg) and b (neg that are truly pos)
a_hat = float(((y == 1) & (T.subtype == "局部脂肪浸潤(無瀰漫字眼)").values).sum() / (y == 1).sum())   # pos that are not diffuse steatosis
# excess of PV L-S<-25 among not_mentioned negatives vs 'no' negatives, x PPV of the criterion, spread over all negatives
b_hat = float(max(0.0, ln["pv_LSlt-25_rate_in_neg_not_mentioned"] - ln["pv_LSlt-25_rate_in_neg_no"])) * (T.hs_cat == "not_mentioned").sum() / (y == 0).sum() + 0.01
def ceil(a, b): return (1 - a) * (1 - b) + 0.5 * ((1 - a) * b + a * (1 - b))
R["ceiling"] = {"a_hat_pos_noise_upper": a_hat, "b_hat_neg_noise": b_hat, "ceiling_noise_model": ceil(a_hat, b_hat),
                "clean_subset_auc_E3": float(A(y[((y == 1) & (T.subtype == "瀰漫/steatosis").values) | (T.hs_cat == "no").values], T.E3_late.values[((y == 1) & (T.subtype == "瀰漫/steatosis").values) | (T.hs_cat == "no").values])),
                "clean_subset_auc_best": float(A(y[((y == 1) & (T.subtype == "瀰漫/steatosis").values) | (T.hs_cat == "no").values], T["E3_late+HU_GBM"].values[((y == 1) & (T.subtype == "瀰漫/steatosis").values) | (T.hs_cat == "no").values])),
                "ceiling_b_only": ceil(0.0, b_hat), "ceiling_a_half": ceil(a_hat / 2, b_hat)}
R["oof"] = {c: float(A(y, T[c])) for c in ["E0", "E3_late"] + list(models)}
json.dump(R, open(O + "summary.json", "w"), indent=1, default=str)
T[["study_id", "fold", "y_hs", "phase_meta", "phase_hu", "subtype", "E0", "E3_late", best]].to_csv(O + "per_case_train_oof.csv", index=False)
print(json.dumps(R["ceiling"], indent=1))
print("ANALYZE_DONE")
