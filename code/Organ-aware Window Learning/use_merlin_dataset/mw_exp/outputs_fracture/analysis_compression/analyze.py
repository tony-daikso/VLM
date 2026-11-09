"""Compression-fracture error analysis on TRAIN OOF (bone_ens). Val used only descriptively at the end."""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from textparse import *
OUT = Path(__file__).parent
d = pd.read_pickle(OUT / "table_trval.pkl")
assert set(d.split) <= {"train", "val"}
d["ypos"] = d.cat.isin(["yes", "probably_yes"]) & (d.comp == "yes")
d["yneg"] = d.cat.isin(["no", "probably_no", "not_mentioned"])
d = d[(d.ypos | d.yneg) & d.bone_ens.notna()].copy()
d["y"] = d.ypos.astype(int)
d["msk"] = d.report.map(msk_section)
d["clv"] = d.quotes.fillna("").map(compression_levels)
for k, v in d.quotes.fillna("").map(flags).apply(pd.Series).add_prefix("q_").items():
    d[k] = v
# negatives: does the musculoskeletal text mention vertebral deformity-like findings?
d["neg_vert_mention"] = d.msk.map(lambda t: bool(NEG_VERT.search(t)))
d["neg_schmorl"] = d.msk.str.contains(r"schmorl", case=False)
d["vcf_sents"] = d.report.map(vcf_mention)
d["neg_compress_word"] = d.vcf_sents.map(len) > 0
d["neg_treated"] = d.report.fillna("").str.contains(r"vertebroplast|kyphoplast", case=False)
d["neg_scoli"] = d.msk.str.contains(r"scolio|kyphos|kyphotic", case=False)
d["neg_hardware"] = d.msk.str.contains(r"spinal fusion|posterior fusion|pedicle screw|spinal hardware|laminectomy|instrumentation|fixation rod|interbody", case=False)
d["neg_degen"] = d.msk.str.contains(r"degenerat|spondylo|osteophyt|scolio|disc space", case=False)
d["neg_osteopenia"] = d.report.fillna("").str.contains(r"osteopeni|osteopor|demineraliz", case=False)
d["neg_mets"] = d.msk.str.contains(r"metasta|lytic|sclerotic lesion|myeloma", case=False)
LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
def lvl_geom(r, lv):
    a, m, p = r.get(f"{lv}_h_ant"), r.get(f"{lv}_h_mid"), r.get(f"{lv}_h_post")
    i = LV.index(lv); nb = [r.get(f"{LV[j]}_h_post") for j in (i - 1, i + 1) if 0 <= j < 10]
    nb = [x for x in nb if pd.notna(x) and x > 0]
    if pd.isna(p) or p <= 0:
        return np.nan
    ref = max([p] + nb)  # Genant-like: min(ant, mid) vs the larger of own/neighbour posterior height
    return 1 - min(a, m) / ref
d["genant_like_max"] = d.apply(lambda r: np.nanmax([lvl_geom(r, l) for l in LV] + [np.nan]) if r.n_levels > 0 else np.nan, axis=1)

T = d[d.split == "train"].copy()
pos, neg = T[T.y == 1], T[T.y == 0]
def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
base = auc(T.y, T.bone_ens)
thr = np.quantile(neg.bone_ens, 0.90)
T["fn"] = (T.y == 1) & (T.bone_ens < thr); T["fp"] = (T.y == 0) & (T.bone_ens >= thr)
res = {"n_pos": len(pos), "n_neg": len(neg), "auc_bone_ens_train_oof": base, "thr_spec90": thr,
       "sens_at_spec90": float((pos.bone_ens >= thr).mean()),
       "auc_geom_maxloss": auc(T.y, T.max_height_loss.fillna(0)), "auc_geom_minap": auc(T.y, -T.min_ant_post_ratio.fillna(1)),
       "auc_genant_like": auc(T.y, T.genant_like_max.fillna(0)), "auc_age": auc(T.y, T.Age.fillna(T.Age.median()))}
print(json.dumps(res, indent=1, default=float))

# ---- positive-side subgroups (each vs ALL negatives)
P = T[T.y == 1].copy()
P["n_lv_named"] = P.clv.map(len)
P["lv_cov"] = P.clv.map(lambda l: [x for x in l if x in LV])
P["lv_unc"] = P.clv.map(lambda l: [x for x in l if x not in LV])
P["cov_status"] = np.select([P.n_lv_named == 0, P.lv_cov.map(len) == 0, P.lv_unc.map(len) > 0], ["未寫椎節", "全部在T8–L5外", "部分在範圍外"], "全部在T8–L5內")
P["lv_seg_missing"] = P.apply(lambda r: any(not bool(r.get(f"P_{x}", False)) for x in r.lv_cov), axis=1)
P["lv_geom_missing"] = P.apply(lambda r: any(pd.isna(r.get(f"{x}_h_post")) for x in r.lv_cov), axis=1)
P["rep_genant"] = P.apply(lambda r: np.nanmax([lvl_geom(r, x) for x in r.lv_cov] + [np.nan]) if r.lv_cov else np.nan, axis=1)
P["sev_kw"] = np.select([P.q_moderate_severe, P.q_mild], ["中重度字眼", "輕度字眼"], "未註明程度")
P["chron_kw"] = np.select([P.acute.isin(["yes", "mixed"]), P.q_chronic | (P.acute == "no")], ["急性/混合", "陳舊/穩定"], "未註明")

negsc = neg.bone_ens.values
def sub_auc(s):
    s = np.asarray(s)
    if len(s) == 0: return np.nan
    return float(((s[:, None] > negsc[None]).mean(1) + 0.5 * (s[:, None] == negsc[None]).mean(1)).mean())
P["pair_auc"] = ((P.bone_ens.values[:, None] > negsc[None]).mean(1))
def cost(mask):
    """AUROC points lost because this positive subgroup ranks worse than the remaining positives
    = n_S/n_pos * (AUC_rest - AUC_S); also fraction of FN at spec90."""
    mask = np.asarray(mask, bool)
    if mask.sum() == 0: return dict(n=0)
    a_s, a_r = P.pair_auc[mask].mean(), P.pair_auc[~mask].mean()
    return dict(n=int(mask.sum()), pct_pos=100 * mask.mean(), auc_sub=a_s, auc_rest=a_r,
                cost_pts=100 * mask.mean() * (a_r - a_s), pct_of_FN=100 * (mask & T.loc[P.index, "fn"].values).sum() / T.loc[P.index, "fn"].sum(),
                sens90=float((P.bone_ens[mask] >= thr).mean()))
rows = []
def add(group, name, mask): rows.append(dict(side="陽性", group=group, subgroup=name, **cost(mask)))
for v in ["全部在T8–L5內", "部分在範圍外", "全部在T8–L5外", "未寫椎節"]: add("H1 椎節覆蓋", v, P.cov_status == v)
add("H1 椎節覆蓋", "含T1–T7", P.lv_unc.map(lambda l: any(x.startswith("T") for x in l)))
add("H1 椎節覆蓋", "n_levels<10", P.n_levels < 10)
add("H1 椎節覆蓋", "n_levels<=6", P.n_levels <= 6)
add("H2 分割", "報告椎節在分割中缺失(P=0)", P.lv_seg_missing)
add("H2 分割", "報告椎節幾何量不到(<150 voxel)", P.lv_geom_missing)
for v in ["輕度字眼", "未註明程度", "中重度字眼"]: add("H3 嚴重度(字眼)", v, P.sev_kw == v)
for v in ["急性/混合", "陳舊/穩定", "未註明"]: add("H3 時期", v, P.chron_kw == v)
add("H3 治療過", "椎體成形術/骨水泥", P.q_treated)
add("H3 型態", "只寫終板(endplate/Schmorl)", P.q_endplate_only & ~P.q_moderate_severe & ~P.quotes.str.contains("compression|wedg|height", case=False))
add("H4 標籤雜訊(疑似)", "後方附件(橫突/棘突/椎弓)", P.q_posterior_elem & ~P.quotes.str.contains("compression|wedg|height|vertebral bod", case=False))
add("H4 標籤雜訊(疑似)", "L5後側楔形(多為正常變異)", P.q_wedge_l5_post)
add("H4 標籤雜訊(疑似)", "猶豫字眼(possible/likely/versus…)", P.q_hedge)
add("H4 標籤雜訊(疑似)", "probably_yes", P.cat == "probably_yes")
add("H4 標籤雜訊(疑似)", "LLM p<80", P.p.astype(float) < 80)
gb = pd.cut(P.rep_genant, [-1, 0.10, 0.20, 0.25, 0.40, 2], labels=["<10%", "10–20%", "20–25%", "25–40%", ">40%"])
for v in gb.cat.categories: add("H5 報告椎節高度流失(3mm幾何)", v, gb == v)
add("H5 報告椎節高度流失(3mm幾何)", "無法量測", P.rep_genant.isna())
ab = pd.cut(P.Age, [0, 50, 65, 75, 200], labels=["<50", "50–65", "65–75", "≥75"])
for v in ab.cat.categories: add("H6 年齡(陽性)", v, ab == v)
pos_tab = pd.DataFrame(rows)

# ---- negative-side subgroups: cost = n_S/n_neg*(AUC_vs_restneg - AUC_vs_S)
ps = pos.bone_ens.values
neg2 = T[T.y == 0].copy()
neg2["pair_auc"] = 1 - ((ps[:, None] > neg2.bone_ens.values[None]).mean(0))  # fraction of positives this negative beats
nrows = []
def addn(group, name, mask):
    mask = np.asarray(mask, bool)
    if mask.sum() == 0: return
    a_s, a_r = 1 - neg2.pair_auc[mask].mean(), 1 - neg2.pair_auc[~mask].mean()
    nrows.append(dict(side="陰性", group=group, subgroup=name, n=int(mask.sum()), pct_neg=100 * mask.mean(), auc_sub=a_s, auc_rest=a_r,
                      cost_pts=100 * mask.mean() * (a_r - a_s), pct_of_FP=100 * (mask & neg2.index.isin(T.index[T.fp])).sum() / T.fp.sum(),
                      fpr90=float((neg2.bone_ens[mask] >= thr).mean())))
addn("H4 漏標(疑似)", "報告提到壓迫/楔形/高度流失(未否定)", neg2.neg_compress_word)
addn("H4 漏標(疑似)", "報告提到 Schmorl", neg2.neg_schmorl)
addn("H4 漏標(疑似)", "幾何:高度流失>25%", neg2.max_height_loss > 0.25)
addn("H4 漏標(疑似)", "幾何:前/後高比<0.75", neg2.min_ant_post_ratio < 0.75)
addn("H4 漏標(疑似)", "幾何:Genant式流失>25%", neg2.genant_like_max > 0.25)
addn("H4 漏標(疑似)", "not_mentioned", neg2.cat == "not_mentioned")
addn("H3 混淆", "椎體成形術/骨水泥", neg2.neg_treated)
addn("H3 混淆", "脊椎內固定/融合", neg2.neg_hardware)
addn("H3 混淆", "脊柱側彎/後凸", neg2.neg_scoli)
addn("H3 混淆", "退化性變化", neg2.neg_degen)
addn("H3 混淆", "骨質疏鬆/骨量減少字眼", neg2.neg_osteopenia)
addn("H3 混淆", "骨轉移/溶骨病灶", neg2.neg_mets)
nab = pd.cut(neg2.Age, [0, 50, 65, 75, 200], labels=["<50", "50–65", "65–75", "≥75"])
for v in nab.cat.categories: addn("H6 年齡(陰性)", v, nab == v)
addn("H1 椎節覆蓋", "n_levels<10", neg2.n_levels < 10)
neg_tab = pd.DataFrame(nrows)

# ---- H6 age
def matched_auc(y, p, age, tol=2.0):
    pos_, neg_ = np.where(y == 1)[0], np.where(y == 0)[0]; num = den = 0.0
    for i in pos_:
        nb = neg_[np.abs(age[neg_] - age[i]) <= tol]
        num += (p[i] > p[nb]).sum() + 0.5 * (p[i] == p[nb]).sum(); den += len(nb)
    return num / den
Ta = T[T.Age.notna()]
age_res = {"age_matched_auc_bone_ens": matched_auc(Ta.y.values, Ta.bone_ens.values, Ta.Age.values),
           "age_matched_auc_genant": matched_auc(Ta.y.values, Ta.genant_like_max.fillna(0).values, Ta.Age.values),
           "auc_age_only": auc(Ta.y, Ta.Age)}
strata = []
for v in ab.cat.categories:
    m = pd.cut(Ta.Age, [0, 50, 65, 75, 200], labels=ab.cat.categories) == v
    strata.append(dict(age=v, n_pos=int(Ta.y[m].sum()), n_neg=int((1 - Ta.y[m]).sum()), auc=auc(Ta.y[m], Ta.bone_ens[m]), auc_genant=auc(Ta.y[m], Ta.genant_like_max[m].fillna(0))))
# residualize: bone_ens AUROC after removing age (logit ~ age on negatives)
print(json.dumps(age_res, indent=1, default=float)); print(pd.DataFrame(strata))

# ---- severity: AUC by reported-level geometric loss for different scorers
sev = []
for v in gb.cat.categories:
    m = (gb == v).values
    if m.sum() < 5: continue
    s = P.genant_like_max.fillna(0)[m].values; ng = neg.genant_like_max.fillna(0).values
    sev.append(dict(bin=v, n=int(m.sum()), auc_bone_ens=sub_auc(P.bone_ens[m]),
                    auc_genant_max=float(((s[:, None] > ng[None]).mean(1) + 0.5 * (s[:, None] == ng[None]).mean(1)).mean())))
print(pd.DataFrame(sev))

# ---- negatives: distribution of geometry, quantization evidence
quant = {"h_values_unique_T12": sorted(T.T12_h_mid.dropna().unique().tolist())[:15],
         "neg_frac_maxloss_gt25": float((neg.max_height_loss > 0.25).mean()), "pos_frac_maxloss_gt25": float((pos.max_height_loss > 0.25).mean()),
         "neg_frac_genant_gt25": float((neg.genant_like_max > 0.25).mean()), "pos_frac_genant_gt25": float((pos.genant_like_max > 0.25).mean()),
         "n_levels_pos_mean": float(pos.n_levels.mean()), "n_levels_neg_mean": float(neg.n_levels.mean()),
         "n_levels_lt10_pos": float((pos.n_levels < 10).mean()), "n_levels_lt10_neg": float((neg.n_levels < 10).mean())}
print(json.dumps(quant, indent=1, default=float))

pd.concat([pos_tab, neg_tab]).to_csv(OUT / "subgroups_train_oof.csv", index=False)
pd.DataFrame(strata).to_csv(OUT / "age_strata_train_oof.csv", index=False)
pd.DataFrame(sev).to_csv(OUT / "severity_train_oof.csv", index=False)
json.dump({**res, **age_res, **quant}, open(OUT / "summary_train_oof.json", "w"), indent=1, ensure_ascii=False, default=float)
keep_cols = ["split", "cat", "comp", "acute", "p", "Age", "n_levels", "bone_ens", "max_height_loss", "min_ant_post_ratio", "genant_like_max"]
P_out = P[keep_cols + ["cov_status", "sev_kw", "chron_kw", "rep_genant", "lv_seg_missing", "pair_auc"]].copy(); P_out["levels"] = P.clv.map(",".join)
P_out.to_csv(OUT / "train_positives_annot.csv")
T.to_pickle(OUT / "train_annot.pkl"); P.to_pickle(OUT / "train_pos_annot.pkl"); neg2.to_pickle(OUT / "train_neg_annot.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(pos_tab.round(3).to_string()); print(neg_tab.round(3).to_string())
print(P.cov_status.value_counts()); print(pd.Series([x for l in P.clv for x in l]).value_counts().head(25))
