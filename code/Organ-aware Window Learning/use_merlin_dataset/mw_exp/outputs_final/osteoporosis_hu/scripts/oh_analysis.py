"""影像骨質疏鬆: label QC, measurement reliability, contrast-phase check, report agreement, and val evaluation of
(b) the embeddings-only model and (c) the report-based osteopenia model against the new label. Train/val only.
usage: python oh_analysis.py WORK_DIR LABELS_CSV GEOM15_LEVELS_CSV MODEL_DIR OUT_DIR"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
W, LAB, GEOM, MD, OUT = [a.rstrip("/") + ("/" if i in (0, 3, 4) else "") for i, a in enumerate(sys.argv[1:6])]
sys.path.insert(0, W)
from ib_core import auc, age_matched_auc, sens_at_spec, boot_ci, spearman_neg
R = {}

def icc(a, b):
    """ICC(2,1) absolute agreement and ICC(3,1) consistency for two raters."""
    X = np.c_[a, b]; n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum(); mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)), float((msr - mse) / (msr + (k - 1) * mse))

def kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int); po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); return float((po - pe) / (1 - pe))

def agree(a, b, name):
    m = ~np.isnan(a) & ~np.isnan(b); a, b = a[m], b[m]; d = a - b
    i2, i3 = icc(a, b)
    r = dict(pair=name, n=int(m.sum()), pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]),
             icc_abs=i2, icc_cons=i3, bias=float(d.mean()), sd_diff=float(d.std()), loa_lo=float(d.mean() - 1.96 * d.std()),
             loa_hi=float(d.mean() + 1.96 * d.std()), mad=float(np.median(np.abs(d))), p_absdiff_gt30=float((np.abs(d) > 30).mean()),
             kappa110=kappa(a <= 110, b <= 110), agree110=float(((a <= 110) == (b <= 110)).mean()),
             kappa160=kappa(a <= 160, b <= 160), agree160=float(((a <= 160) == (b <= 160)).mean()))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r

L = pd.read_csv(LAB).set_index("study_id"); L = L[L.split.isin(["train", "val"])]
meta = pd.read_csv(W + "bundle_trval.csv", usecols=["study_id", "Age", "Gender", "contrast", "phase"]).set_index("study_id")
L = L.join(meta)
rep = pd.read_csv(W + "labels_trval.csv", usecols=["study_id", "split", "y_op", "op_cat", "y_frac"]).set_index("study_id")
rep = rep[rep.split.isin(["train", "val"])]
L = L.join(rep[["y_op", "op_cat", "y_frac"]])
lab = L[L.hu.notna()]

# ---------------------------------------------------------------- A. counts
cnt = []
for sp, g in L.groupby("split"):
    gl = g[g.hu.notna()]
    c = dict(split=sp, n=len(g), roi_seg15=int((g.roi_src == "seg15_roi").sum()), roi_geom15_fallback=int((g.roi_src == "geom15_fallback").sum()),
             no_measurement=int((g.exclude == "no_measurement").sum()),
             cement_excl=int((g.exclude == "cement").sum()), cement_report=int(g.cement_rep.sum()), cement_hu=int(g.cement_hu.sum()),
             cement_both=int((g.cement_rep & g.cement_hu).sum()),
             L1_invalid=int((~g.L1_valid.astype(bool)).sum()), L1_frac=int(g.L1_frac.sum()), L1_frac_report=int(g.L1_frac_rep.sum()),
             L1_frac_geom=int(g.L1_frac_geo.sum()), L1_scl=int(g.L1_scl.sum()), L1_scl_focal=int(g.L1_scl_focal.sum()),
             L1_scl_abs=int(g.L1_scl_abs.sum()), L1_scl_rel=int(g.L1_scl_rel.sum()), L1_scl_report=int(g.L1_scl_rep.sum()),
             fb_L1_invalid=int((g.fallback_reason == "L1_invalid").sum()), fb_L1_fracture=int((g.fallback_reason == "L1_fracture").sum()),
             fb_L1_sclerosis=int((g.fallback_reason == "L1_sclerosis").sum()), used_L2=int((g.level_used == "L2").sum()),
             no_valid_level=int((g.exclude == "no_valid_level").sum()), labelled=len(gl), used_L1=int((gl.level_used == "L1").sum()),
             y110=int(gl.y110.sum()), y160=int(gl.y160.sum()), prev110=float(gl.y110.mean()), prev160=float(gl.y160.mean()),
             cat_osteoporosis=int((gl.cat3 == "骨質疏鬆").sum()), cat_osteopenia=int((gl.cat3 == "骨量減少").sum()), cat_normal=int((gl.cat3 == "正常").sum()),
             hardware_report=int(g.hardware_rep.sum()))
    cnt.append(c)
R["counts"] = cnt; pd.DataFrame(cnt).to_csv(OUT + "counts.csv", index=False); print(pd.DataFrame(cnt).T)
age = []
for (sp, c3), g in lab.groupby(["split", "cat3"]):
    age.append(dict(split=sp, cat3=c3, n=len(g), age_med=g.Age.median(), age_q1=g.Age.quantile(.25), age_q3=g.Age.quantile(.75),
                    pct_ge70=(g.Age >= 70).mean(), pct_lt50=(g.Age < 50).mean(), pct_female=(g.Gender == "Female").mean(), hu_med=g.hu.median()))
for sp, g in lab.groupby("split"):
    b = pd.cut(g.Age, [0, 50, 60, 70, 80, 200], right=False)
    for k, gg in g.groupby(b, observed=True):
        age.append(dict(split=sp, cat3=f"age {k}", n=len(gg), prev110=gg.y110.mean(), prev160=gg.y160.mean(), hu_med=gg.hu.median(),
                        pct_female=(gg.Gender == "Female").mean()))
pd.DataFrame(age).to_csv(OUT + "age_by_category.csv", index=False)

# ---------------------------------------------------------------- B. HU QC
roi = pd.read_csv(W + "roi_levels.csv"); roi = roi[roi.id.isin(L.index)]
R["roi_status"] = roi.groupby("level").status.value_counts().unstack(fill_value=0).to_dict(orient="index")
ok = roi[roi.status == "ok"]
R["roi_erosion_small_frac"] = ok.groupby("level").erosion.apply(lambda s: float((s == "small").mean())).to_dict()
R["roi_n_median"] = ok.groupby("level").n_roi.median().to_dict()
R["label_hu_pct"] = {sp: g.hu.quantile([.001, .01, .05, .25, .5, .75, .95, .99, .999]).round(1).to_dict() for sp, g in lab.groupby("split")}
R["label_hu_outliers"] = {sp: dict(lt0=int((g.hu < 0).sum()), gt350=int((g.hu > 350).sum()), lt30=int((g.hu < 30).sum())) for sp, g in lab.groupby("split")}
o1 = ok[ok.level == "L1"]
R["leakage"] = dict(L1_roi_sd_median=float(o1.sd.median()), L1_p95_minus_med_median=float((o1.p95 - o1.med).median()),
                    L1_f400_gt5pct=float((o1.f400 > .05).mean()))
geo = pd.read_csv(GEOM, usecols=["id", "level", "hu_body"]); geo = geo[geo.id.isin(L.index)]
gp = geo.pivot(index="id", columns="level", values="hu_body")
rp = ok.pivot(index="id", columns="level", values="med")

# ---------------------------------------------------------------- C. reliability
rel = []
cl = L[(L.roi_src == "seg15_roi") & (L.cement == 0)]
both = cl[(cl.L1_valid.astype(bool)) & (cl.L2_valid.astype(bool)) & (cl.L1_frac == 0) & (cl.L2_frac == 0) & (cl.L1_scl == 0) & (cl.L2_scl == 0)]
rel.append(agree(both.L1_hu.values, both.L2_hu.values, "L1 vs L2 (1.5 mm trabecular ROI; both valid, no fx/sclerosis)"))
ids_ = rp.index.intersection(gp.index)
rel.append(agree(rp.loc[ids_, "L1"].values, gp.loc[ids_, "L1"].values, "L1 trabecular ROI vs geom15 hu_body (same 1.5 mm, 2 px erosion)"))
h3 = pd.concat([pd.read_csv(W + f, usecols=["study_id", "aorta_hu", "L1_core_med", "L1_core_n"]) for f in ("hu_ct.csv", "hu_a6.csv")]).drop_duplicates("study_id").set_index("study_id")
h3 = h3[h3.index.isin(L.index)]
ids3 = rp.index.intersection(h3.index)
rel.append(agree(rp.loc[ids3, "L1"].values, h3.loc[ids3, "L1_core_med"].values.astype(float), "L1 1.5 mm trabecular ROI vs 3 mm core HU (TotalSeg 3 mm mask, eroded core)"))
pd.DataFrame(rel).to_csv(OUT + "reliability.csv", index=False); R["reliability"] = rel
R["invalid_rate"] = {sp: dict(L1_invalid=float(1 - g.L1_valid.astype(bool).mean()), L2_invalid=float(1 - g.L2_valid.astype(bool).mean()),
                              unlabelled=float(g.hu.isna().mean())) for sp, g in L.groupby("split")}

# ---------------------------------------------------------------- D. contrast phase
ph = lab.join(h3[["aorta_hu"]])
ph["aorta_grp"] = pd.cut(ph.aorta_hu, [-1e9, 100, 250, 1e9], labels=["aorta<100（近似無顯影）", "100–250（門靜脈期）", "≥250（動脈期為主）"])
def ols(y, X, names):
    """OLS with intercept; returns {name: (coef, ci_lo, ci_hi)} (normal-approx 95 % CI)."""
    X = np.c_[np.ones(len(y)), X]; beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = res @ res / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {n: dict(coef=float(beta[i + 1]), ci=[float(beta[i + 1] - 1.96 * se[i + 1]), float(beta[i + 1] + 1.96 * se[i + 1])])
            for i, n in enumerate(names)} | dict(n=int(len(y)))
d = ph.dropna(subset=["Age"]).assign(female=lambda x: (x.Gender == "Female").astype(float))
d = d[d.contrast.notna()]
base = np.c_[d.Age.values, d.Age.values ** 2, d.female.values]
phase = dict(meta_phase=d.phase.value_counts(dropna=False).to_dict(), meta_contrast=d.contrast.value_counts(dropna=False).to_dict(),
             aorta_available=int(d.aorta_hu.notna().sum()))
phase["by_meta_contrast"] = d.groupby("contrast").agg(n=("hu", "size"), hu_med=("hu", "median"), age_med=("Age", "median")).round(1).reset_index().to_dict(orient="records")
phase["by_meta_contrast_aorta_med"] = d.groupby("contrast").aorta_hu.median().round(1).to_dict()
phase["by_aorta_grp"] = d.groupby("aorta_grp", observed=True).agg(n=("hu", "size"), hu_med=("hu", "median"), age_med=("Age", "median"),
                                                                   prev110=("y110", "mean"), prev160=("y160", "mean")).round(3).reset_index().astype(str).to_dict(orient="records")
phase["ols_contrast_flag"] = ols(d.hu.values, np.c_[base, d.contrast.astype(bool).astype(float).values], ["age", "age2", "female", "contrast_true"])["contrast_true"]
da = d.dropna(subset=["aorta_hu"]); bA = np.c_[da.Age.values, da.Age.values ** 2, da.female.values]
g = da.aorta_grp.astype(str).values
phase["ols_aorta_grp_vs_lt100"] = ols(da.hu.values, np.c_[bA, (g == "100–250（門靜脈期）").astype(float), (g == "≥250（動脈期為主）").astype(float)],
                                       ["age", "age2", "female", "venous", "arterial"])
phase["ols_aorta_slope_per100HU"] = ols(da.hu.values, np.c_[bA, da.aorta_hu.values / 100], ["age", "age2", "female", "aorta100"])["aorta100"]
R["phase"] = phase; print(json.dumps(phase, ensure_ascii=False, indent=0, default=str)[:3000])

# ---------------------------------------------------------------- E. report agreement
ra = []
for sp, g in lab.groupby("split"):
    g = g.dropna(subset=["y_op"])
    ct = pd.crosstab(g.cat3, g.op_cat)
    ct.to_csv(OUT + f"crosstab_cat3_vs_report_{sp}.csv")
    ra.append(dict(split=sp, n=len(g), report_pos=int(g.y_op.sum()), y110=int(g.y110.sum()),
                   p_reportpos_given_y110=float(g.y_op[g.y110 == 1].mean()), p_reportpos_given_y160=float(g.y_op[g.y160 == 1].mean()),
                   p_reportpos_given_normal=float(g.y_op[g.cat3 == "正常"].mean()),
                   p_y110_given_reportpos=float(g.y110[g.y_op == 1].mean()), p_y160_given_reportpos=float(g.y160[g.y_op == 1].mean()),
                   kappa_y110_report=kappa(g.y110, g.y_op), kappa_y160_report=kappa(g.y160, g.y_op),
                   auc_negHU_for_report=auc(g.y_op.values, -g.hu.values),
                   hu_med_reportpos=float(g.hu[g.y_op == 1].median()), hu_med_reportneg=float(g.hu[g.y_op == 0].median())))
R["report_agreement"] = ra; pd.DataFrame(ra).to_csv(OUT + "report_agreement.csv", index=False); print(pd.DataFrame(ra).T)

# ---------------------------------------------------------------- F. val evaluation
V = lab[lab.split == "val"]; T = lab[lab.split == "train"]
ev = []
def bin_eval(name, tgt, s_val, s_oof=None, ids_val=None):
    g = V.loc[ids_val]; y = g[tgt].values.astype(float); s = np.asarray(s_val, float)
    r = dict(model=name, target=tgt, n=len(y), n_pos=int(y.sum()), auc=auc(y, s), ci=boot_ci(y, s),
             age_matched=age_matched_auc(y, s, g.Age.values), sens90=sens_at_spec(y, s)[0],
             spearman_neg_age=spearman_neg(y, s, g.Age.values))
    if s_oof is not None:
        gT = T.loc[s_oof.index]; r["oof_auc"] = auc(gT[tgt].values, s_oof.values)
    ev.append(r); print(name, tgt, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("model", "target")}, flush=True)

mi = json.load(open(MD + "model_info.json"))
reg_v = pd.read_csv(MD + "val_hu_reg.csv").set_index("study_id"); reg_o = pd.read_csv(MD + "oof_hu_reg.csv").set_index("study_id")
yv = V.loc[reg_v.index, "hu"].values; pv = reg_v["stack"].values
regr = dict(n=len(yv), mae=float(np.mean(np.abs(yv - pv))), rmse=float(np.sqrt(np.mean((yv - pv) ** 2))), pearson=float(pearsonr(yv, pv)[0]),
            icc_abs=icc(yv, pv)[0], bias=float((pv - yv).mean()), oof_mae=mi["hu_reg"]["oof_mae"],
            oof_pearson=float(pearsonr(T.loc[reg_o.index, "hu"].values, reg_o["stack"].values)[0]))
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    ix = rng.integers(0, len(yv), len(yv)); bs.append((np.mean(np.abs(yv[ix] - pv[ix])), pearsonr(yv[ix], pv[ix])[0], icc(yv[ix], pv[ix])[0]))
bs = np.array(bs); regr.update(mae_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), pearson_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                               icc_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist())
for k, gsub in (("L1_used", V.loc[reg_v.index].level_used == "L1"), ("L2_fallback", V.loc[reg_v.index].level_used == "L2"),
                ("age>=65", V.loc[reg_v.index].Age >= 65)):
    m = gsub.values; regr[f"mae_{k}"] = float(np.mean(np.abs(yv[m] - pv[m]))); regr[f"pearson_{k}"] = float(pearsonr(yv[m], pv[m])[0]); regr[f"n_{k}"] = int(m.sum())
for k, c in mi["hu_reg"]["components"].items():
    regr[f"comp_{k}_val_mae"] = float(np.mean(np.abs(yv - reg_v[k].values)))
R["regression"] = regr; print("regression", regr)
for tgt in ("y110", "y160"):
    sv = pd.read_csv(MD + f"val_{tgt}.csv").set_index("study_id"); so = pd.read_csv(MD + f"oof_{tgt}.csv").set_index("study_id")
    bin_eval("嵌入模型（分類 stack）", tgt, sv["stack"].values, so["stack"], sv.index)
    bin_eval("嵌入模型（−預測 HU）", tgt, -reg_v["stack"].reindex(sv.index).values, -reg_o["stack"], sv.index)
    for c in [c for c in sv.columns if c != "stack"]:
        bin_eval(f"元件 {c}", tgt, sv[c].values, so[c], sv.index)
    # chosen by train OOF
    ch = mi[tgt]["chosen"]; R[f"chosen_{tgt}"] = ch
    # (c) report-based osteopenia models vs the new label
    A = pd.read_csv(W + "A_val.csv").set_index("study_id"); Ao = pd.read_csv(W + "A_oof.csv").set_index("study_id")
    iv = V.index.intersection(A.index); io = T.index.intersection(Ao.index)
    bin_eval("報告標籤模型 A（osteopenia_labels 重跑）", tgt, A.loc[iv, "stack"].values, Ao.loc[io, "stack"], iv)
    fs = pd.read_csv(W + "stack_val.csv").set_index("study_id"); fo = pd.read_csv(W + "stack_oof_train.csv").set_index("study_id")
    iv2 = V.index.intersection(fs.index); io2 = T.index.intersection(fo.index)
    bin_eval("正式報告模型 5win+v15（.890）", tgt, fs.loc[iv2, "5win+v15"].values, fo.loc[io2, "5win+v15"], iv2)
    # same-subset comparison of the chosen embedding model and the report model
    ivc = sv.index.intersection(A.index)
    bin_eval("嵌入模型（分類 stack）＠與報告模型同一子集", tgt, sv.loc[ivc, "stack"].values, None, ivc)
    bin_eval("報告標籤模型 A＠同一子集", tgt, A.loc[ivc, "stack"].values, None, ivc)
A = pd.read_csv(W + "A_val.csv").set_index("study_id"); iv = V.index.intersection(A.index)
R["report_model_spearman_with_hu_val"] = float(spearmanr(A.loc[iv, "stack"], V.loc[iv, "hu"])[0])
R["embed_reg_spearman_with_hu_val"] = float(spearmanr(reg_v["stack"], V.loc[reg_v.index, "hu"])[0])
pd.DataFrame(ev).to_csv(OUT + "val_results.csv", index=False)
# paired delta embedding vs report model on y110 / y160
from ib_core import boot_delta
dl = {}
for tgt in ("y110", "y160"):
    sv = pd.read_csv(MD + f"val_{tgt}.csv").set_index("study_id"); ivc = sv.index.intersection(A.index); y = V.loc[ivc, tgt].values
    d = boot_delta(y, A.loc[ivc, "stack"].values, sv.loc[ivc, "stack"].values); dl[tgt] = d
R["delta_embed_minus_reportmodel"] = dl
json.dump(R, open(OUT + "summary.json", "w"), indent=1, ensure_ascii=False, default=str)
print("ANALYSIS_DONE")
