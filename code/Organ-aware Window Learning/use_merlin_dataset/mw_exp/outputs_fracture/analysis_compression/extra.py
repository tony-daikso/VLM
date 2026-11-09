"""Exclusive FN/FP categories at spec90, joint mild x age, clean-subset AUROCs, noise ceiling, localization (per-level OOF), val descriptive."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from textparse import *
OUT = Path(__file__).parent
T = pd.read_pickle(OUT / "train_annot.pkl"); P = pd.read_pickle(OUT / "train_pos_annot.pkl"); N = pd.read_pickle(OUT / "train_neg_annot.pkl")
assert (T.split == "train").all()
S = json.load(open(OUT / "summary_train_oof.json")); thr = S["thr_spec90"]
LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
ns = N.bone_ens.values
def pa(s):
    s = np.asarray(s, float); return float(((s[:, None] > ns[None]).mean(1) + 0.5 * (s[:, None] == ns[None]).mean(1)).mean()) if len(s) else np.nan
q = P.quotes.fillna("")
P["f_noise"] = (P.cat == "probably_yes") | P.q_hedge | q.str.contains(r"developmental|limbus|normal variant|physiolog", case=False) \
    | (P.q_posterior_elem & ~q.str.contains("compression|wedg|height|vertebral bod", case=False)) | P.q_wedge_l5_post
P["f_endplate"] = P.q_endplate_only & ~P.q_moderate_severe & ~q.str.contains("compression|wedg|height", case=False)
P["f_cov"] = P.cov_status.isin(["全部在T8–L5外"]) | P.lv_seg_missing
P["f_mild"] = P.q_mild & ~P.q_moderate_severe
P["f_young"] = P.Age < 50
fn = P[P.bone_ens < thr]
cat_fn = np.select([fn.f_cov, fn.f_noise, fn.f_endplate, fn.f_mild & fn.f_young, fn.f_mild, fn.f_young, fn.chron_kw == "陳舊/穩定"],
                   ["範圍外/分割缺失", "疑似標籤雜訊(猶豫/發育性/可能)", "只有終板變化", "輕度＋<50歲", "輕度(≥50歲)", "<50歲(非輕度)", "陳舊/穩定(非輕度)"], "其他(中重度或未註明)")
fn_tab = pd.Series(cat_fn).value_counts().rename("n").to_frame(); fn_tab["pct_FN"] = 100 * fn_tab.n / len(fn)
# category AUROC & cost using full positive set categorisation
cat_all = np.select([P.f_cov, P.f_noise, P.f_endplate, P.f_mild & P.f_young, P.f_mild, P.f_young, P.chron_kw == "陳舊/穩定"],
                    ["範圍外/分割缺失", "疑似標籤雜訊(猶豫/發育性/可能)", "只有終板變化", "輕度＋<50歲", "輕度(≥50歲)", "<50歲(非輕度)", "陳舊/穩定(非輕度)"], "其他(中重度或未註明)")
P["cat_x"] = cat_all
best = P.pair_auc[P.cat_x == "其他(中重度或未註明)"].mean()
g = P.groupby("cat_x").agg(n=("pair_auc", "size"), auc=("pair_auc", "mean"), sens90=("bone_ens", lambda s: (s >= thr).mean()))
g["pct_pos"] = 100 * g.n / len(P); g["cost_vs_clean_pts"] = g.pct_pos * (best - g.auc)
fn_tab = fn_tab.join(g, rsuffix="_pos")
print(fn_tab.round(3).to_string())

fp = N[N.bone_ens >= thr]
def ncat(D):
    return np.select([D.neg_compress_word | D.neg_treated, D.neg_hardware | D.neg_scoli, D.neg_osteopenia, D.neg_mets, D["cat"] == "not_mentioned",
                      D.neg_degen & (D.Age >= 65), D.Age >= 75, D.neg_degen, D.Age < 18],
                     ["報告有壓迫/椎體成形字眼(漏標)", "內固定/側彎/後凸(結構混淆)", "骨質疏鬆字眼", "骨轉移/溶骨", "報告未提骨骼(not_mentioned)",
                      "退化+≥65歲", "≥75歲(無其他)", "退化(<65歲)", "<18歲"], "其他")
N["cat_x"] = ncat(N)
fp_tab = pd.Series(ncat(fp)).value_counts().rename("n").to_frame(); fp_tab["pct_FP"] = 100 * fp_tab.n / len(fp)
ps = P.bone_ens.values
N["pair_auc"] = 1 - ((ps[:, None] > N.bone_ens.values[None]).mean(0) + 0.5 * (ps[:, None] == N.bone_ens.values[None]).mean(0))
best_n = N.pair_auc[N.cat_x == "其他"].mean()
gn = N.groupby("cat_x").agg(n=("pair_auc", "size"), auc=("pair_auc", lambda s: 1 - s.mean()), fpr90=("bone_ens", lambda s: (s >= thr).mean()))
gn["pct_neg"] = 100 * gn.n / len(N); gn["cost_vs_clean_pts"] = gn.pct_neg * ((1 - best_n) - gn.auc)
fp_tab = fp_tab.join(gn, how="outer", rsuffix="_neg").fillna(0).sort_values("n", ascending=False)
print(fp_tab.round(3).to_string())

# joint mild x age
jt = []
for mild in (True, False):
    for lo, hi, nm in [(0, 50, "<50"), (50, 65, "50–65"), (65, 75, "65–75"), (75, 200, "≥75")]:
        m = (P.f_mild == mild) & (P.Age >= lo) & (P.Age < hi)
        mn = (N.Age >= lo) & (N.Age < hi)
        s = P.bone_ens[m].values; nn = N.bone_ens[mn].values
        a_in = float(((s[:, None] > nn[None]).mean(1)).mean()) if len(s) else np.nan
        jt.append(dict(mild=mild, age=nm, n_pos=int(m.sum()), auc_vs_all_neg=pa(s), auc_vs_same_age_neg=a_in))
jt = pd.DataFrame(jt); print(jt.round(3).to_string())

# clean-subset AUROCs (what the model does on unambiguous labels)
def A(pm, nm):
    y = np.r_[np.ones(pm.sum()), np.zeros(nm.sum())]; s = np.r_[P.bone_ens[pm], N.bone_ens[nm]]; return roc_auc_score(y, s)
allp, alln = np.ones(len(P), bool), np.ones(len(N), bool)
clean = {
    "all": A(allp, alln),
    "pos−疑似雜訊": A(~P.f_noise.values, alln),
    "pos−疑似雜訊−終板only": A(~(P.f_noise | P.f_endplate).values, alln),
    "pos−疑似雜訊−終板−輕度": A(~(P.f_noise | P.f_endplate | P.f_mild).values, alln),
    "neg−漏標/內固定/側彎": A(allp, ~(N.neg_compress_word | N.neg_treated | N.neg_hardware | N.neg_scoli).values),
    "neg−not_mentioned": A(allp, (N["cat"] != "not_mentioned").values),
    "兩邊都清(去雜訊+終板+漏標+內固定+not_mentioned)": A(~(P.f_noise | P.f_endplate).values, ~(N.neg_compress_word | N.neg_treated | N.neg_hardware | N.neg_scoli | (N["cat"] == "not_mentioned")).values),
}
print(json.dumps(clean, indent=1, ensure_ascii=False))
# noise ceiling formula: perfect model on true labels, a = frac positives wrong, b = frac negatives truly positive
def ceil(a, b): return (1 - a) * (1 - b) + 0.5 * ((1 - a) * b + a * (1 - b))
def realistic(auc_true, a, b):  # wrong positives score like negatives (0.5), hidden positives among neg score like positives
    return (1 - a) * ((1 - b) * auc_true + b * 0.5) + a * ((1 - b) * 0.5 + b * (1 - auc_true))
noise = [dict(a=a, b=b, perfect_model_ceiling=ceil(a, b), auc_if_true_auc_0p95=realistic(0.95, a, b), auc_if_true_auc_0p93=realistic(0.93, a, b))
         for a in (0.02, 0.05, 0.08) for b in (0.005, 0.01, 0.02, 0.03)]
noise = pd.DataFrame(noise); print(noise.round(3).to_string())

# localization from per-level OOF (vertebra head, 3 seeds)
loc = {}
f = OUT / "perunit_oof.npz"
if f.exists():
    z = np.load(f, allow_pickle=True)
    U = pd.DataFrame(z["vert_lvl"], index=z["ids"], columns=LV); assert set(z["split"]) <= {"train", "val"}
    U = U.loc[U.index.isin(T.index)]
    loc["perunit_any_auc_train"] = roc_auc_score(T.y.loc[U.index], pd.Series(z["logit_any"], index=z["ids"]).loc[U.index])
    Pp = P.join(U, how="inner")
    Pp = Pp[Pp.lv_cov.map(len) > 0]
    top = Pp[LV].idxmax(axis=1)
    hit = [t in l for t, l in zip(top, Pp.lv_cov)]
    hit1 = [any(abs(LV.index(t) - LV.index(x)) <= 1 for x in l) for t, l in zip(top, Pp.lv_cov)]
    rk = [min([v for v in (r[LV].rank(ascending=False))[l].values if v == v] or [99]) for (_, r), l in zip(Pp.iterrows(), Pp.lv_cov)]
    nlv = Pp[LV].notna().sum(1)
    chance = np.mean([len([x for x in l if not np.isnan(r[x])]) / max(n, 1) for (_, r), l, n in zip(Pp.iterrows(), Pp.lv_cov, nlv)])
    Pp["hit"] = hit; Pp["hit1"] = hit1
    loc.update(n=len(Pp), top1_hit=float(np.mean(hit)), top1_within1=float(np.mean(hit1)), chance_top1=float(chance), median_rank_reported=float(np.median(rk)))
    for nm, m in [("TP(≥thr)", Pp.bone_ens >= thr), ("FN(<thr)", Pp.bone_ens < thr), ("輕度", Pp.f_mild), ("非輕度", ~Pp.f_mild)]:
        loc[f"top1_hit_{nm}"] = float(Pp.hit[m].mean()); loc[f"n_{nm}"] = int(m.sum())
    # peakiness: max - median of level logits; positives vs FP negatives vs TN
    pk = (U.max(1) - U.median(1)).rename("peak")
    Tp = T.join(pk, how="inner")
    loc["peak_TP"] = float(Tp.peak[(Tp.y == 1) & (Tp.bone_ens >= thr)].median())
    loc["peak_FN"] = float(Tp.peak[(Tp.y == 1) & (Tp.bone_ens < thr)].median())
    loc["peak_FP"] = float(Tp.peak[(Tp.y == 0) & (Tp.bone_ens >= thr)].median())
    loc["peak_TN"] = float(Tp.peak[(Tp.y == 0) & (Tp.bone_ens < thr)].median())
    # FP negatives whose peak is as focal as the median TP: candidate unreported fractures
    loc["FP_with_TPlike_peak_frac"] = float((Tp.peak[(Tp.y == 0) & (Tp.bone_ens >= thr)] >= loc["peak_TP"]).mean())
    loc["neg_with_TPlike_peak_and_high_score_frac_of_neg"] = float(((Tp.y == 0) & (Tp.bone_ens >= thr) & (Tp.peak >= loc["peak_TP"])).sum() / (Tp.y == 0).sum())
    # per-level location distribution of reported positives
    lvl_counts = pd.Series([x for l in Pp.lv_cov for x in l]).value_counts().reindex(LV).fillna(0).astype(int)
    hit_by_lvl = {lv: float(np.mean([h for h, l in zip(hit, Pp.lv_cov) if lv in l])) if lvl_counts[lv] else np.nan for lv in LV}
    loc["hit_by_level"] = hit_by_lvl; loc["n_by_level"] = lvl_counts.to_dict()
    top_fp = Tp[(Tp.y == 0) & (Tp.bone_ens >= thr)].join(U, rsuffix="_u")
    loc["FP_top_level_dist"] = top_fp[LV].idxmax(axis=1).value_counts().to_dict()
    loc["TP_top_level_dist"] = Tp[(Tp.y == 1) & (Tp.bone_ens >= thr)].join(U, rsuffix="_u")[LV].idxmax(axis=1).value_counts().to_dict()
print(json.dumps(loc, indent=1, ensure_ascii=False, default=float))

fn_tab.to_csv(OUT / "fn_categories_spec90_train.csv"); fp_tab.to_csv(OUT / "fp_categories_spec90_train.csv")
jt.to_csv(OUT / "mild_x_age_train.csv", index=False); noise.to_csv(OUT / "noise_ceiling.csv", index=False)
json.dump(dict(clean=clean, loc=loc), open(OUT / "extra_train.json", "w"), indent=1, ensure_ascii=False, default=float)
