"""Rib-only decomposition, age matching, pooling comparison, pelvis analysis (TRAIN OOF)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from analyze_util import auc
from parse import pelv_info
OUT = Path(__file__).parent
d = pd.read_pickle(OUT / "table_parsed.pkl")
tr = d[d.split == "train"].copy()
rng = np.random.default_rng(1)
res = {}
neg = tr[tr.y == 0]
def A(pos, s, negdf=neg): return round(auc(np.r_[np.ones(len(pos)), np.zeros(len(negdf))], np.r_[pos[s], negdf[s]]), 4)
ro = tr[(tr.rib == 1) & ~tr.other_site_rib]; rw = tr[(tr.rib == 1) & tr.other_site_rib]
SC = ["s_bone_ens", "s_stack", "s_Gsite_rib", "s_Gsite_vert", "s_Gsite_pelv", "s_ref__site_max__mean__all3", "s_ref__site_max__max__all3", "s_e3late", "Age"]
res["rib_only_vs_neg"] = {s: A(ro, s) for s in SC}; res["rib_with_other_vs_neg"] = {s: A(rw, s) for s in SC}
res["n_rib_only"], res["n_rib_with_other"] = len(ro), len(rw)
res["other_sites_in_rib_pos"] = tr[tr.rib == 1].sites.value_counts().head(8).to_dict()
# age-matched (+-2y) AUROC: each positive matched to up to 10 negatives
def age_matched(pos, s, k=10, B=3):
    out = []
    na = neg.Age.values
    for b in range(B):
        sel = []
        for a in pos.Age.values:
            c = np.where(np.abs(na - a) <= 2)[0]
            sel += list(rng.choice(c, min(k, len(c)), replace=False))
        nn = neg.iloc[sel]
        out.append(auc(np.r_[np.ones(len(pos)), np.zeros(len(nn))], np.r_[pos[s], nn[s]]))
    return round(float(np.mean(out)), 4)
pr = tr[tr.rib == 1]
res["age_matched_rib"] = {s: age_matched(pr, s) for s in ["s_bone_ens", "s_Gsite_rib", "s_ref__site_max__mean__all3", "s_ref__site_max__max__all3"]}
res["age_matched_rib_only"] = {s: age_matched(ro, s) for s in ["s_bone_ens", "s_Gsite_rib"]}
# paired bootstrap: max vs mean pooling (rib target)
pp, nn_ = pr, neg
def pb(s1, s2, pos, B=500):
    out = []
    for _ in range(B):
        p = pos.sample(len(pos), replace=True, random_state=int(rng.integers(1e9))); n = nn_.sample(len(nn_), replace=True, random_state=int(rng.integers(1e9)))
        yy = np.r_[np.ones(len(p)), np.zeros(len(n))]
        out.append(auc(yy, np.r_[p[s1], n[s1]]) - auc(yy, np.r_[p[s2], n[s2]]))
    return [round(float(np.mean(out)), 4), round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
res["max_minus_mean_rib"] = pb("s_ref__site_max__max__all3", "s_ref__site_max__mean__all3", pr)
res["max_minus_mean_rib_only"] = pb("s_ref__site_max__max__all3", "s_ref__site_max__mean__all3", ro)
res["G_ribsite_minus_boneens_rib_only"] = pb("s_Gsite_rib", "s_bone_ens", ro)
# rib-only subtype AUROCs
rows = []
for c in ["rib_age", "llm_acute", "rib_mult", "rib_disp", "rib_loc", "rib_level_band", "fov", "trunc"]:
    for g, gdf in ro.groupby(c, dropna=False):
        if len(gdf) >= 8:
            rows.append({"變項": c, "子群": str(g), "陽性數(僅肋骨)": len(gdf), "AUROC bone_ens": A(gdf, "s_bone_ens"), "AUROC G肋骨site": A(gdf, "s_Gsite_rib")})
pd.DataFrame(rows).to_csv(OUT / "subgroup_auroc_rib_only.csv", index=False)
print(pd.DataFrame(rows).to_string())
# --------------------- pelvis ---------------------
pv = tr[tr.pelv == 1].copy()
res["n_pelv"] = len(pv)
def pel_sub(r):
    s = r.sites
    if r.p_sacral_insuff or (r.p_insuff and r.p_sacrum): return "薦椎不全骨折"
    if r.p_insuff: return "其他不全/壓力骨折"
    return None
pv["pel_kind"] = pv.apply(lambda r: pel_sub(r) or ("急性外傷" if r.p_acute and not r.p_old else "陳舊/癒合" if r.p_old else "未描述時序"), axis=1)
pv["pel_loc"] = np.select([pv.p_sacrum & ~(pv.p_rami | pv.p_acet | pv.p_iliac | pv.p_femur), pv.p_rami & ~(pv.p_sacrum | pv.p_acet | pv.p_femur),
                           pv.p_femur & ~(pv.p_sacrum | pv.p_rami | pv.p_acet | pv.p_iliac), pv.p_coccyx & ~(pv.p_sacrum | pv.p_rami | pv.p_femur)],
                          ["僅薦椎", "僅恥骨枝", "僅股骨/髖", "僅尾椎"], "多處/其他")
pv["pel_hw"] = np.where(pv.p_hardware, "有植入物/關節置換", "無")
pv["pel_only"] = np.where(pv.other_site_pelv, "合併他處骨折", "僅骨盆")
fv = pv.v_femur_l.notna()
pv["femur_cov"] = np.where(~fv, None, np.where(pv[["v_femur_l", "v_femur_r"]].min(1) < 500, "股骨覆蓋少（<500 vox）", "股骨覆蓋足"))
pv["pel_conf"] = np.where(pv.cat == "probably_yes", "probably_yes", np.where(pv.p_uncert, "報告含不確定字樣", "明確"))
pv["acute_llm"] = pv.acute
negd = neg.copy()
negd["femur_cov"] = np.where(negd.v_femur_l.isna(), None, np.where(negd[["v_femur_l", "v_femur_r"]].min(1) < 500, "股骨覆蓋少（<500 vox）", "股骨覆蓋足"))
negd["hw"] = negd.report.isna() if "report" in negd else None
rows = []
for c in ["pel_kind", "pel_loc", "pel_hw", "pel_only", "femur_cov", "pel_conf", "acute_llm"]:
    for g, gdf in pv.groupby(c, dropna=False):
        if len(gdf) >= 5:
            nd = negd[negd.femur_cov == g] if c == "femur_cov" else neg
            rows.append({"變項": c, "子群": str(g), "陽性數": len(gdf), "AUROC bone_ens": A(gdf, "s_bone_ens", nd), "AUROC G骨盆site": A(gdf, "s_Gsite_pelv", nd)})
pt = pd.DataFrame(rows); pt.to_csv(OUT / "subgroup_auroc_pelvis.csv", index=False); print(pt.to_string())
res["pelv_overall"] = {s: A(pv, s) for s in ["s_bone_ens", "s_Gsite_pelv", "s_e3late", "Age"]}
res["pelv_age_matched"] = {"s_bone_ens": None}
# pelvis error table @90% spec
thr = np.quantile(neg.s_bone_ens, 0.9); fnm = pv.s_bone_ens < thr
et = []
for name, m in [("植入物/關節置換（偽影或術後）", pv.p_hardware), ("尾椎/恥骨枝等小骨（僅此）", pv.pel_loc.isin(["僅尾椎", "僅恥骨枝"])),
                ("陳舊/癒合", pv.pel_kind == "陳舊/癒合"), ("股骨覆蓋少", pv.femur_cov == "股骨覆蓋少（<500 vox）"),
                ("報告不確定", pv.pel_conf != "明確"), ("其餘", pd.Series(True, index=pv.index))]:
    m = m.fillna(False).astype(bool) & ~pd.Series([any(x) for x in zip(*[r for r in et_masks])] if (et_masks := [e[1] for e in et]) else [False] * len(pv), index=pv.index)
    et.append((name, m))
rows = [{"類別（依序互斥）": n, "陽性 n": int(m.sum()), "陽性占比": f"{m.mean():.1%}", "敏感度@90%特異度": f"{(~fnm[m]).mean():.1%}" if m.sum() else "-",
         "FN n": int((m & fnm).sum()), "占 FN %": f"{(m & fnm).sum() / fnm.sum():.1%}"} for n, m in et]
pd.DataFrame(rows).to_csv(OUT / "error_table_pelvis_bone_ens.csv", index=False); print(pd.DataFrame(rows).to_string())
res["pelv_sens90"] = float((~fnm).mean()); res["pelv_hw_rate_pos"] = float(pv.p_hardware.mean())
res["neg_hw_rate"] = float(neg.p_hardware.mean()) if "p_hardware" in neg else None
res["pelv_hw_neg_fp_rate"] = float((neg[neg.p_hardware].s_bone_ens >= thr).mean())
res["femur_cov_pos"] = pv.femur_cov.value_counts(dropna=False).to_dict()
json.dump(res, open(OUT / "results_rib2.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
