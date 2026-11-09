"""Rib / pelvic fracture error analysis on TRAIN OOF (val descriptive only). Writes csv + results.json."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from parse import rib_info, neg_rib_mention, pelv_info
OUT = Path(__file__).parent
d = pd.read_pickle(OUT / "table_trval.pkl")
assert set(d.split) <= {"train", "val"}
d = d[d.s_bone_ens.notna()].copy()
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}
d["y"] = np.where(d.cat.isin(POS), 1.0, np.where(d.cat.isin(NEG), 0.0, np.nan))
has = lambda k: d.sites.fillna("").str.split("|").apply(lambda s: k in s)
d["rib"] = np.where(d.y == 0, 0.0, np.where((d.y == 1) & has("rib"), 1.0, np.nan))
d["pelv"] = np.where(d.y == 0, 0.0, np.where((d.y == 1) & (has("pelvis") | has("sacrum") | has("hip_femur")), 1.0, np.nan))
d["other_site_rib"] = (d.rib == 1) & d.sites.fillna("").str.contains("vertebra|pelvis|sacrum|hip_femur|other")
d["other_site_pelv"] = (d.pelv == 1) & d.sites.fillna("").str.contains("vertebra|rib|other")
R = pd.DataFrame([rib_info(t) for t in d.report], index=d.index)
N = pd.DataFrame([neg_rib_mention(t) for t in d.report], index=d.index)
P = pd.DataFrame([pelv_info(t) for t in d.report], index=d.index)
d = d.join(R).join(N).join(P)

def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~np.isnan(y) & ~np.isnan(s); y, s = y[m], s[m]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

rng = np.random.default_rng(0)
def auc_ci(sp, sn, B=400):
    sp, sn = np.asarray(sp, float), np.asarray(sn, float)
    if len(sp) < 5: return (auc(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]), np.nan, np.nan)
    a = auc(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn])
    bs = []
    for _ in range(B):
        p = rng.choice(sp, len(sp)); n = rng.choice(sn, min(len(sn), 3000))
        bs.append(auc(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

tr = d[d.split == "train"]; va = d[d.split == "val"]
res = {}
SC = ["s_bone_ens", "s_stack", "s_Gsite_rib", "s_G_any", "s_ref__site_max__mean__all3", "s_ref__site_max__max__all3",
      "s_ref__site_max__mean__bone", "s_ref__site_lse__mean__all3", "s_e3late", "s_ens_all", "s_Gsite_pelv", "Age"]
res["overall"] = {t: {s: round(auc(tr[t], tr[s]), 4) for s in SC} for t in ["rib", "pelv", "y"]}
res["overall_val_descr"] = {t: {s: round(auc(va[t], va[s]), 4) for s in ["s_bone_ens", "s_Gsite_rib", "s_Gsite_pelv"]} for t in ["rib", "pelv"]}
res["n_train"] = {"rib_pos": int((tr.rib == 1).sum()), "pelv_pos": int((tr.pelv == 1).sum()), "neg": int((tr.y == 0).sum())}

# ---------- FOV of reported rib(s): rib-unit presence (>=100 vox) from emb_ribs for all ids ----------
def unit_of(lv): return "1_4" if lv <= 4 else str(lv)
def fov_status(r):
    if not r.rib_levels or r.rib_emb_missing: return "未標明肋骨節次" if not r.rib_levels else "無肋骨嵌入"
    st = []
    for tok in r.rib_levels.split(";"):
        sd, lv = tok[0], int(tok[1:])
        sides = ["left", "right"] if sd in "B?" else [{"L": "left", "R": "right"}[sd]]
        st.append(any(bool(r[f"PR_rib_{s}_{unit_of(lv)}"]) for s in sides))
    return "報告肋骨全在 FOV" if all(st) else "報告肋骨全不在 FOV" if not any(st) else "部分在 FOV"
d["fov"] = [fov_status(r) for r in d.itertuples()] if False else d.apply(fov_status, axis=1)
d.loc[d.rib != 1, "fov"] = None
# truncation (A6000 masks subset): any reported rib touching the superior/inferior slab edge
def trunc(r):
    if r.rib != 1 or not r.rib_levels or np.isnan(r.nz): return None
    t = []
    for tok in r.rib_levels.split(";"):
        sd, lv = tok[0], int(tok[1:])
        for s in (["left", "right"] if sd in "B?" else [{"L": "left", "R": "right"}[sd]]):
            v = r[f"v_rib_{s}_{lv}"]
            t.append("缺" if v < 100 else "截斷" if (r.get(f"top_rib_{s}_{lv}", 0) == 1) else "完整")
    return "完整" if all(x == "完整" for x in t) else "全缺" if all(x == "缺" for x in t) else "截斷/部分"
d["trunc"] = d.apply(trunc, axis=1)
ribv = [f"v_rib_{s}_{i}" for s in ("left", "right") for i in range(1, 13)]
d["rib_vox_total"] = d[ribv].sum(1, min_count=1)
d["n_rib_units_present"] = d[[c for c in d.columns if c.startswith("PR_")]].sum(1)
d["ribs_vox_present"] = (d[ribv] >= 100).sum(1).where(d.nz.notna())
lung_up = d[["v_lung_ul_l", "v_lung_ul_r"]].sum(1, min_count=1)
d["coverage"] = np.where(d.nz.isna(), None,
                 np.where(d.v_T4.fillna(0) >= 100, "胸腹（含 T4 以上）",
                 np.where(d.v_T8.fillna(0) >= 100, "上腹延伸（含 T8，不到 T4）", "腹部（無 T8）")))
d["cov_units"] = pd.cut(d.n_rib_units_present, [-1, 6, 10, 14, 18], labels=["≤6 單元", "7–10", "11–14", "15–18"])
tr = d[d.split == "train"]

S = "s_bone_ens"
neg_tr = tr[tr.y == 0]
def sub_rows(df, target, col, score=S, negdf=None, within=False):
    rows = []
    for g, gdf in df[df[target] == 1].groupby(col, dropna=False):
        nd = negdf if negdf is not None else (df[(df.y == 0) & (df[col] == g)] if within else df[df.y == 0])
        a, lo, hi = auc_ci(gdf[score].values, nd[score].values)
        rows.append({"變項": col, "子群": str(g), "陽性數": len(gdf), "陰性數": len(nd), "AUROC": round(a, 3), "95%CI": f"[{lo:.3f}, {hi:.3f}]"})
    return rows

sub = []
sub += sub_rows(tr, "rib", "fov")
sub += sub_rows(tr, "rib", "trunc")
sub += sub_rows(tr, "rib", "coverage", within=True)
sub += sub_rows(tr, "rib", "cov_units", within=True)
sub += sub_rows(tr, "rib", "other_site_rib")
# subtype
def age_grp(r):
    if r.t_acute and not (r.t_old or r.t_healing): return "急性"
    if r.t_healing and not r.t_acute: return "癒合中/亞急性"
    if r.t_old and not r.t_acute: return "陳舊/已癒合/變形"
    if r.t_acute and (r.t_old or r.t_healing): return "混合"
    return "未描述"
d["rib_age"] = d.apply(age_grp, axis=1).where(d.rib == 1)
d["rib_disp"] = np.where(d.t_disp, "移位/粉碎", np.where(d.t_nondisp, "無移位/輕微", "未描述")); d.loc[d.rib != 1, "rib_disp"] = None
d["rib_mult"] = np.where((d.n_levels >= 2) | d.t_multi, "多發", np.where(d.n_levels == 1, "單一", "未明")); d.loc[d.rib != 1, "rib_mult"] = None
d["rib_loc"] = np.where(d.t_post & ~d.t_ant, "後側", np.where(d.t_ant & ~d.t_post, "前側", np.where(d.t_lat & ~d.t_ant & ~d.t_post, "外側", np.where(d.t_ant | d.t_post | d.t_lat, "多處", "未描述"))))
d.loc[d.rib != 1, "rib_loc"] = None
d["rib_level_band"] = np.where(d.lv_max.isna(), "未標明", np.where(d.lv_max <= 6, "僅 1–6 肋", np.where(d.lv_min >= 9, "僅 9–12 肋", "含 7–8 肋")))
d.loc[d.rib != 1, "rib_level_band"] = None
d["lab_conf"] = np.where(d.cat == "probably_yes", "probably_yes", np.where(d.p.fillna(100) < 85, "yes, p<85", "yes, p≥85")); d.loc[d.rib != 1, "lab_conf"] = None
d["llm_acute"] = d.acute.where(d.rib == 1)
d["age_band"] = pd.cut(d.Age, [0, 50, 65, 75, 120], labels=["<50", "50–65", "65–75", "≥75"])
d["uncert"] = d.t_uncert.where(d.rib == 1)
d["patho"] = d.t_patho.where(d.rib == 1)
tr = d[d.split == "train"]
for c in ["rib_age", "llm_acute", "rib_disp", "rib_mult", "rib_loc", "rib_level_band", "lab_conf", "uncert", "patho"]:
    sub += sub_rows(tr, "rib", c)
sub += sub_rows(tr, "rib", "age_band", within=True)
# G rib-site score for key subgroups
for c in ["fov", "rib_age", "coverage"]:
    sub += [dict(r, 變項=r["變項"] + " [G 肋骨 site 分數]") for r in sub_rows(tr, "rib", c, score="s_Gsite_rib", within=(c == "coverage"))]
subdf = pd.DataFrame(sub)
subdf.to_csv(OUT / "subgroup_auroc_rib.csv", index=False)

# ---------- error table @90% specificity ----------
def err_table(df, target, score, fn_cats, fp_cats, tag):
    neg = df[df.y == 0]; pos = df[df[target] == 1]
    thr = np.quantile(neg[score], 0.90)
    fn = pos[pos[score] < thr]; fp = neg[neg[score] >= thr]
    rows = []
    rem = pd.Series(True, index=pos.index)
    for name, f in fn_cats:
        m = f(pos) & rem
        rem &= ~m
        rows.append({"類型": "FN", "類別（依序互斥）": name, "陽性中 n": int(m.sum()), "陽性占比": f"{m.mean():.1%}",
                     "此類敏感度@90%特異度": f"{(pos[m][score] >= thr).mean():.1%}" if m.sum() else "-",
                     "FN n": int((m & (pos[score] < thr)).sum()), "占 FN %": f"{(m & (pos[score] < thr)).sum() / len(fn):.1%}"})
    rem = pd.Series(True, index=neg.index)
    for name, f in fp_cats:
        m = f(neg) & rem
        rem &= ~m
        rows.append({"類型": "FP", "類別（依序互斥）": name, "陽性中 n": int(m.sum()), "陽性占比": f"{m.mean():.1%}（陰性中）",
                     "此類敏感度@90%特異度": f"FP 率 {(neg[m][score] >= thr).mean():.1%}" if m.sum() else "-",
                     "FN n": int((m & (neg[score] >= thr)).sum()), "占 FN %": f"{(m & (neg[score] >= thr)).sum() / len(fp):.1%}（占 FP）"})
    t = pd.DataFrame(rows); t.to_csv(OUT / f"error_table_{tag}.csv", index=False)
    return t, dict(thr=float(thr), sens=float((pos[score] >= thr).mean()), n_fn=len(fn), n_fp=len(fp), n_pos=len(pos), n_neg=len(neg))

FN_R = [("報告肋骨不在 FOV（全部節次缺）", lambda x: x.fov == "報告肋骨全不在 FOV"),
        ("報告肋骨部分在 FOV", lambda x: x.fov == "部分在 FOV"),
        ("無肋骨嵌入/分割", lambda x: x.fov == "無肋骨嵌入"),
        ("陳舊/已癒合/變形（無急性字樣）", lambda x: x.rib_age == "陳舊/已癒合/變形"),
        ("標註不確定（probably_yes / 報告含可能、疑似）", lambda x: (x.cat == "probably_yes") | x.t_uncert),
        ("無移位/輕微（含癒合中）", lambda x: (x.rib_disp == "無移位/輕微") | (x.rib_age == "癒合中/亞急性")),
        ("節次未標明（模糊描述）", lambda x: x.fov == "未標明肋骨節次"),
        ("其餘：在 FOV、明確描述", lambda x: pd.Series(True, index=x.index))]
FP_R = [("報告提及肋骨陳舊骨折/變形（疑似漏標）", lambda x: x.nr_old_frx | x.nr_deform),
        ("報告提及其他骨折字樣未否定（他處陳舊骨折等）", lambda x: x.nr_frx_word_unneg | x.report.str.contains(r"(?i)(old|healed|remote|chronic)[^.]{0,30}fractur")),
        ("骨病灶（溶骨/硬化/轉移）", lambda x: x.report.str.contains(r"(?i)lytic|sclerotic lesion|metasta[^.]{0,40}(bone|osseous|skelet)|osseous metast|myeloma")),
        ("骨質疏鬆標註陽性", lambda x: x.osteopenia.isin(["yes", "probably_yes"])),
        ("高齡 ≥75 歲", lambda x: x.Age >= 75),
        ("其餘", lambda x: pd.Series(True, index=x.index))]
et_rib, ei_rib = err_table(tr, "rib", S, FN_R, FP_R, "rib_bone_ens")
et_ribG, ei_ribG = err_table(tr, "rib", "s_Gsite_rib", FN_R, FP_R, "rib_Gsite")
res["err_rib"] = ei_rib; res["err_ribG"] = ei_ribG

# ---------- label consistency ----------
oldtxt = d.report.str.contains(r"(?i)(old|healed|remote|chronic|healing)[^.]{0,40}\bribs?\b[^.]{0,30}fractur|(old|healed|remote|chronic|healing)[^.]{0,20}rib fractur|\bribs?\b[^.]{0,40}(fracture deformit|deformit)|rib fractures?[^.]{0,30}(old|healed|remote|chronic)")
lc = d[d.split == "train"].assign(oldtxt=oldtxt).groupby("oldtxt").apply(lambda g: pd.Series({
    "n": len(g), "rib_pos": int((g.rib == 1).sum()), "neg": int((g.y == 0).sum()), "unclear_NaN": int(g.y.isna().sum()),
    "pos_other_site_only": int(((g.y == 1) & (g.rib != 1)).sum())}))
res["label_consistency_oldribtext"] = lc.to_dict()
tr = d[d.split == "train"]
res["neg_with_oldrib_text"] = {"n": int((tr.y == 0)[oldtxt.loc[tr.index]].sum()),
                               "mean_score_pct": float(((tr[S].rank(pct=True))[(tr.y == 0) & oldtxt.loc[tr.index]]).mean())}
res["unclear_with_rib"] = int(((tr.cat == "unclear") & tr.report.str.contains(r"(?i)\bribs?\b[^.]{0,40}fractur")).sum())
# cleaned-label AUROC: drop negatives with old-rib text
cl = tr[~((tr.y == 0) & (tr.nr_old_frx | tr.nr_deform))]
res["auc_rib_drop_suspect_negs"] = round(auc(cl.rib, cl[S]), 4)
res["n_suspect_negs"] = int(((tr.y == 0) & (tr.nr_old_frx | tr.nr_deform)).sum())

# ---------- FOV aggregates / ceiling ----------
pos = tr[tr.rib == 1]
res["fov_dist"] = pos.fov.value_counts().to_dict()
res["trunc_dist"] = pos.trunc.value_counts().to_dict()
res["level_hist"] = pd.Series([int(t[1:]) for s in pos.rib_levels for t in s.split(";") if t]).value_counts().sort_index().to_dict()
res["n_pos_with_levels"] = int((pos.rib_levels != "").sum())
res["rib_age_dist"] = pos.rib_age.value_counts().to_dict()
res["llm_acute_dist"] = pos.acute.value_counts().to_dict()
res["other_site_rib_frac"] = float(pos.other_site_rib.mean())
# rib presence in positives vs negatives
res["rib_units_present_mean"] = {"pos": float(pos.n_rib_units_present.mean()), "neg": float(tr[tr.y == 0].n_rib_units_present.mean())}
res["rib_vox_total_median"] = {"pos": float(pos.rib_vox_total.median()), "neg": float(tr[tr.y == 0].rib_vox_total.median())}
res["coverage_dist_pos"] = pos.coverage.value_counts().to_dict(); res["coverage_dist_neg"] = tr[tr.y == 0].coverage.value_counts().to_dict()
d.drop(columns=["report"]).to_pickle(OUT / "table_parsed.pkl")
d.loc[d.rib == 1, ["split", "cat", "p", "acute", "sites", "rib_levels", "side", "fov", "trunc", "rib_age", "rib_disp", "rib_mult", "rib_loc",
                   "rib_level_band", "coverage", "n_rib_units_present", "Age", S, "s_Gsite_rib"]].to_csv(OUT / "rib_positives_parsed.csv")
json.dump(res, open(OUT / "results_rib.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(subdf.to_string())
print(et_rib.to_string()); print(ei_rib)
print(et_ribG.to_string()); print(ei_ribG)
