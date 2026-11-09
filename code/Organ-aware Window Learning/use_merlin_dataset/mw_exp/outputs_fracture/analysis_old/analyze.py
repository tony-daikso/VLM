"""Old (chronic/healed) fracture error analysis on TRAIN OOF. Val is reported descriptively only (nothing tuned on it)."""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
OUT = MW / "outputs_fracture/analysis_old"
sys.path.insert(0, str(MW / "src")); sys.path.insert(1, str(MW / "outputs_fracture/analysis_compression"))
from age_shortcut import matched_auc
from textparse import compression_levels, msk_section, COV

d = pd.read_pickle(OUT / "table_old.pkl")
assert set(d.split) <= {"train", "val"}
POSC, NEGC = ["yes", "probably_yes"], ["no", "probably_no", "not_mentioned"]
d = d[d.cat.isin(POSC + NEGC) & d.s_bone_ens.notna() & d.Age.notna()].copy()
d["y"] = d.cat.isin(POSC).astype(int)
d["grp"] = np.where(d.y == 0, "neg", d.acute.map({"no": "old", "yes": "acute", "mixed": "acute", "not_stated": "not_stated"}).fillna("not_stated"))
GN = {"old": "陳舊", "acute": "急性(含混合)", "not_stated": "未註明", "all": "全部骨折"}
S = "s_bone_ens"
q = d.quotes.fillna("").str.lower()
rep = d.report.fillna("")
msk = rep.map(msk_section)

# ---------- report-text flags ----------
SUB = {
    "healed": r"\bhealed|\bhealing",
    "remote": r"\bremote",
    "chronic": r"\bchronic",
    "old": r"\bold\b",
    "unchanged_prior": r"unchanged|\bstable|again (?:seen|noted|demonstrated)|redemonstrat|similar to|compared to prior|from prior|\bknown\b|previously",
    "deformity": r"deformit",
    "hardware": r"fixation|\borif\b|screws?\b|\brods?\b|\bplate\b|hardware|(?<!ef)fusion|arthroplast|prosthe|intramedullary|\bnail|instrumentation|laminectomy",
    "cement": r"vertebroplast|kyphoplast|cement|augmentation",
    "age_indet": r"age.indeterminate|indeterminate (?:age|chronicity)|of uncertain age|undetermined age",
    "hedged": r"possibl|probabl|may represent|cannot be excluded|questionable|suspect|equivocal|likely|versus|\bvs\b",
}
for k, v in SUB.items():
    d["q_" + k] = q.str.contains(v, regex=True)
d["any_chronic_word"] = d[["q_healed", "q_remote", "q_chronic", "q_old"]].any(axis=1)
d["r_hardware"] = msk.str.contains(SUB["hardware"], case=False, regex=True) | rep.str.contains(r"spinal fusion|pedicle screw|hip arthroplast|orif|intramedullary|dynamic hip screw", case=False, regex=True)
d["r_cement"] = rep.str.contains(r"vertebroplast|kyphoplast", case=False, regex=True)
d["r_degen"] = msk.str.contains(r"degenerat|spondylo|osteophyt|disc (?:space|height|bulg)|facet arthro", case=False, regex=True)
d["r_schmorl"] = rep.str.contains(r"schmorl", case=False)
d["r_scolio"] = rep.str.contains(r"scolio|curvature", case=False)
d["r_osteopenia"] = d.osteopenia.isin(["yes", "probably_yes"]) | rep.str.contains(r"osteopeni|osteopor|demineraliz", case=False, regex=True)
d["r_lesion"] = msk.str.contains(r"metasta|lytic|sclerotic (?:lesion|focus|foci)|myeloma|blastic", case=False, regex=True)
SENT = re.compile(r"(?<=[.;])\s+|\n+")
NEGS = re.compile(r"\bno\b[^.;]{0,40}(?:fractur|compression|deformit)|without[^.;]{0,25}fractur|negative for|no evidence", re.I)
OLDFX = re.compile(r"(?:\bold|healed|healing|remote|chronic|prior|previous|post.?traumatic)\b[^.;]{0,40}(?:fractur|deformit)|fractur[^.;]{0,40}\b(?:old|healed|remote|chronic)|compression (?:deformit|fractur)|wedge (?:compression|deformit)|vertebroplast|kyphoplast|(?:vertebral body|vertebral) height loss|loss of (?:vertebral )?body height", re.I)


def noise_sents(t):
    return [s for s in SENT.split(t) if OLDFX.search(s) and not NEGS.search(s)]


d["neg_frx_sents"] = np.where(d.y == 0, rep.map(lambda t: " | ".join(noise_sents(t))[:300]), "")
d["neg_frx_mention"] = (d.y == 0) & (d.neg_frx_sents != "")
d["neg_frx_rib"] = d.neg_frx_mention & d.neg_frx_sents.str.contains(r"\brib", case=False)
d["neg_frx_vert"] = d.neg_frx_mention & d.neg_frx_sents.str.contains(r"compression|wedge|vertebr|plast", case=False)
d["neg_frx_cement"] = d.neg_frx_mention & d.neg_frx_sents.str.contains(r"plast", case=False)
d["neg_noacute_phrase"] = (d.y == 0) & rep.str.contains(r"no acute (?:osseous|fractur|bony)", case=False)
# sites
for s_ in ["vertebra", "rib", "pelvis", "sacrum", "hip_femur", "other"]:
    d["site_" + s_] = d.sites.fillna("").str.split("|").map(lambda L, s_=s_: s_ in L)
d["site_pelv"] = d.site_pelvis | d.site_sacrum
d["comp_yes"] = d.comp == "yes"
d["n_sites"] = d[["site_vertebra", "site_rib", "site_pelv", "site_hip_femur", "site_other"]].sum(1)
# FOV / segmentation flags
d["clv"] = d.quotes.fillna("").map(compression_levels)
d["vert_out_fov"] = d.site_vertebra & d.clv.map(lambda L: len(L) > 0 and not any(l in COV for l in L))


def rib_fov(r):
    if not isinstance(r.rib_rib_levels, str) or not r.rib_rib_levels:
        return np.nan
    tot, seen = 0, 0
    for tok in r.rib_rib_levels.split(";"):
        sd, lv = tok[0], int(tok[1:])
        sides = ["left", "right"] if sd in "B?" else [{"L": "left", "R": "right"}[sd]]
        for s_ in sides:
            v = r.get(f"v_rib_{s_}_{lv}")
            tot += 1; seen += int(pd.notna(v) and v > 200)
    return seen / tot if tot else np.nan


d["rib_fov_frac"] = d.apply(rib_fov, axis=1)
d["vseg_missing"] = d.n_levels.isna() | (d.n_levels < 6)
d.drop(columns=["clv"]).to_pickle(OUT / "table_old_flags.pkl")

T = d[d.split == "train"].copy(); V = d[d.split == "val"].copy()


def auc(y, s):
    y = np.asarray(y); return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def maucs(D, s=S):
    return matched_auc(D.y.values, D[s].values.astype(float), D.Age.values.astype(float), 2.0)


def sub_vs_neg(D, mask):
    return D[(D.y == 0) | ((D.y == 1) & mask)]


def row(D, mask, name, s=S, matched=True):
    X = sub_vs_neg(D, mask)
    r = dict(group=name, n_pos=int(X.y.sum()), age_pos_mean=X.Age[X.y == 1].mean(), auc_age=auc(X.y, X.Age), auc_model=auc(X.y, X[s]))
    if matched:
        r["auc_model_age_matched"] = maucs(X, s); r["auc_age_age_matched"] = maucs(X, "Age")
    r["gain_over_age"] = r["auc_model"] - r["auc_age"]
    return r


res = {}
neg = T[T.y == 0]
res["n_train_neg"] = len(neg); res["n_train_pos"] = int(T.y.sum())
res["grp_counts_train"] = T.grp.value_counts().to_dict()
res["spearman_score_age_neg"] = {s: float(spearmanr(neg[s], neg.Age, nan_policy="omit")[0]) for s in ["s_bone_ens", "s_ens_all", "s_e3late"]}
res["age_neg_mean"] = float(neg.Age.mean())

# ---- 1. age table
rows = []
for s in ["s_bone_ens", "s_ens_all", "s_e3late"]:
    for g in ["all", "old", "acute", "not_stated"]:
        m = T.y == 1 if g == "all" else T.grp == g
        r = row(T, m, GN[g], s); r["score"] = s; rows.append(r)
t1 = pd.DataFrame(rows); t1.to_csv(OUT / "t1_age_vs_model.csv", index=False)
# bootstrap diff old-acute (model AUROC and model-age gain)
rng = np.random.default_rng(0); bs = []
idx_n, idx_o, idx_a = [np.where((T.grp == g).values)[0] for g in ["neg", "old", "acute"]]
sc, ag = T[S].values, T.Age.values
for b in range(300):
    n_, o_, a_ = rng.choice(idx_n, len(idx_n)), rng.choice(idx_o, len(idx_o)), rng.choice(idx_a, len(idx_a))
    yo = np.r_[np.zeros(len(n_)), np.ones(len(o_))]; ya = np.r_[np.zeros(len(n_)), np.ones(len(a_))]
    mo, ma = roc_auc_score(yo, sc[np.r_[n_, o_]]), roc_auc_score(ya, sc[np.r_[n_, a_]])
    go, ga = mo - roc_auc_score(yo, ag[np.r_[n_, o_]]), ma - roc_auc_score(ya, ag[np.r_[n_, a_]])
    bs.append((mo - ma, go - ga))
bs = np.array(bs)
res["boot_old_minus_acute_model"] = [float(np.percentile(bs[:, 0], q_)) for q_ in (2.5, 50, 97.5)]
res["boot_old_minus_acute_gain_over_age"] = [float(np.percentile(bs[:, 1], q_)) for q_ in (2.5, 50, 97.5)]
# strata
rows = []
for lo, hi, nm in [(0, 50, "<50"), (50, 65, "50–65"), (65, 75, "65–75"), (75, 200, ">75")]:
    St = T[(T.Age >= lo) & (T.Age < hi)]
    for g in ["all", "old", "acute", "not_stated"]:
        m = St.y == 1 if g == "all" else St.grp == g
        X = sub_vs_neg(St, m)
        rows.append(dict(age_stratum=nm, group=GN[g], n_pos=int(X.y.sum()), n_neg=int((X.y == 0).sum()),
                         prev=float(X.y.mean()), auc_model=auc(X.y, X[S]) if X.y.sum() >= 10 else np.nan))
t1b = pd.DataFrame(rows); t1b.to_csv(OUT / "t1b_age_strata.csv", index=False)

# ---- 2. sites
res["site_mix"] = {}
for g in ["old", "acute", "not_stated"]:
    G = T[T.grp == g]
    res["site_mix"][g] = {k: float(G[k].mean()) for k in ["site_vertebra", "comp_yes", "site_rib", "site_pelv", "site_hip_femur", "site_other"]}
    res["site_mix"][g]["multi_site"] = float((G.n_sites > 1).mean()); res["site_mix"][g]["n"] = len(G)
rows = []
SITES = [("site_vertebra", "椎體", "s_site_vert"), ("comp_yes", "椎體壓迫性", "s_site_vert"), ("site_rib", "肋骨", "s_site_rib"),
         ("site_pelv", "骨盆/薦椎", "s_site_pelv"), ("site_hip_femur", "髖/股骨", "s_site_pelv")]
for col, nm, ss in SITES:
    for g in ["old", "acute", "not_stated"]:
        for only in [False, True]:
            m = (T.grp == g) & T[col] & ((T.n_sites == 1) if only else True)
            X = sub_vs_neg(T, m)
            if X.y.sum() < 8:
                continue
            rows.append(dict(site=nm + ("（單一部位）" if only else ""), group=GN[g], n_pos=int(X.y.sum()), auc_model=auc(X.y, X[S]),
                             auc_site_head=auc(X.y, X[ss]) if ss in X and X[ss].notna().all() else np.nan,
                             auc_age=auc(X.y, X.Age), auc_model_age_matched=maucs(X)))
t2 = pd.DataFrame(rows); t2.to_csv(OUT / "t2_site.csv", index=False)

# ---- 3. subtypes (positives with the word in LLM fracture quotes)
rows = []
SUBN = {"healed": "healed 癒合", "remote": "remote 遠期", "chronic": "chronic 慢性", "old": "old 陳舊", "unchanged_prior": "與先前比較不變/穩定",
        "deformity": "deformity 變形", "hardware": "術後/金屬植入物", "cement": "椎體成形術骨水泥", "age_indet": "年代不明", "hedged": "語氣不確定"}
for k, nm in SUBN.items():
    for g in ["old", "acute", "all"]:
        m = (T.y == 1 if g == "all" else T.grp == g) & T["q_" + k]
        X = sub_vs_neg(T, m)
        if X.y.sum() < 8:
            continue
        rows.append(dict(subtype=nm, group=GN[g], n_pos=int(X.y.sum()), frac_of_group=float(m.sum() / ((T.y == 1) if g == "all" else (T.grp == g)).sum()),
                         auc_model=auc(X.y, X[S]), auc_age=auc(X.y, X.Age), auc_model_age_matched=maucs(X)))
t3 = pd.DataFrame(rows); t3.to_csv(OUT / "t3_subtype.csv", index=False)
# confounder: hardware / cement in negatives
conf = {}
for k in ["r_hardware", "r_cement", "r_degen", "r_schmorl", "r_scolio", "r_osteopenia", "r_lesion", "neg_frx_mention"]:
    a_, b_ = neg[neg[k]], neg[~neg[k]]
    conf[k] = dict(n_neg=len(a_), frac_neg=float(len(a_) / len(neg)), auc_flag_neg_vs_other_neg=auc(np.r_[np.ones(len(a_)), np.zeros(len(b_))], np.r_[a_[S], b_[S]]) if len(a_) else np.nan,
                   frac_pos_with_flag=float(T[T.y == 1][k].mean()), frac_old_with_flag=float(T[T.grp == "old"][k].mean()))
res["neg_flag_score_shift"] = conf

# ---- 4. geometry
rows = []
for g in ["old", "acute", "not_stated"]:
    G = T[(T.grp == g) & T.comp_yes & T.max_height_loss.notna()]
    rows.append(dict(group=GN[g] + " 壓迫性", n=len(G), max_height_loss_median=G.max_height_loss.median(),
                     min_ap_ratio_median=G.min_ant_post_ratio.median(), frac_maxloss_lt015=float((G.max_height_loss < 0.15).mean()),
                     frac_ap_ge_085=float((G.min_ant_post_ratio >= 0.85).mean())))
Gn = neg[neg.max_height_loss.notna()]
rows.append(dict(group="陰性", n=len(Gn), max_height_loss_median=Gn.max_height_loss.median(), min_ap_ratio_median=Gn.min_ant_post_ratio.median(),
                 frac_maxloss_lt015=float((Gn.max_height_loss < 0.15).mean()), frac_ap_ge_085=float((Gn.min_ant_post_ratio >= 0.85).mean())))
t4 = pd.DataFrame(rows)
bins = [(-1, 0.15, "<0.15（幾乎正常）"), (0.15, 0.25, "0.15–0.25"), (0.25, 0.35, "0.25–0.35"), (0.35, 9, "≥0.35")]
rows = []
for g in ["old", "acute"]:
    for lo, hi, nm in bins:
        m = (T.grp == g) & T.comp_yes & (T.max_height_loss > lo) & (T.max_height_loss <= hi)
        X = sub_vs_neg(T, m)
        rows.append(dict(group=GN[g] + " 壓迫性", bin_max_height_loss=nm, n_pos=int(X.y.sum()),
                         auc_model=auc(X.y, X[S]) if X.y.sum() >= 8 else np.nan, auc_geom=auc(X.y, X.max_height_loss.fillna(0)) if X.y.sum() >= 8 else np.nan))
    m = (T.grp == g) & T.comp_yes & T.max_height_loss.isna()
    X = sub_vs_neg(T, m); rows.append(dict(group=GN[g] + " 壓迫性", bin_max_height_loss="無幾何（分割缺）", n_pos=int(X.y.sum()), auc_model=auc(X.y, X[S]) if X.y.sum() >= 8 else np.nan))
t4b = pd.DataFrame(rows)
t4.to_csv(OUT / "t4_geometry.csv", index=False); t4b.to_csv(OUT / "t4b_geometry_bins.csv", index=False)
res["auc_geom_old_comp"] = auc(sub_vs_neg(T, (T.grp == "old") & T.comp_yes).y, sub_vs_neg(T, (T.grp == "old") & T.comp_yes).max_height_loss.fillna(0))
res["auc_geom_acute_comp"] = auc(sub_vs_neg(T, (T.grp == "acute") & T.comp_yes).y, sub_vs_neg(T, (T.grp == "acute") & T.comp_yes).max_height_loss.fillna(0))

# ---- 5. label consistency
ln = dict(n_neg=len(neg), neg_frx_mention=int(neg.neg_frx_mention.sum()), neg_frx_rib=int(neg.neg_frx_rib.sum()),
          neg_frx_vert=int(neg.neg_frx_vert.sum()), neg_frx_cement=int(neg.neg_frx_cement.sum()),
          neg_noacute_phrase=int(neg.neg_noacute_phrase.sum()),
          neg_by_cat=neg[neg.neg_frx_mention].cat.value_counts().to_dict())
ln["auc_noisyneg_vs_cleanneg"] = conf["neg_frx_mention"]["auc_flag_neg_vs_other_neg"]
clean = T[~T.neg_frx_mention]
ln["auc_all_excl_noisyneg"] = auc(clean.y, clean[S]); ln["auc_all"] = auc(T.y, T[S])
for g in ["old", "acute"]:
    ln[f"auc_{g}"] = auc(sub_vs_neg(T, T.grp == g).y, sub_vs_neg(T, T.grp == g)[S])
    ln[f"auc_{g}_excl_noisyneg"] = auc(sub_vs_neg(clean, clean.grp == g).y, sub_vs_neg(clean, clean.grp == g)[S])
# old positives with weak evidence: hedged / probably_yes / low p
O = T[T.grp == "old"]
ln["old_probably_yes"] = float((O.cat == "probably_yes").mean()); ln["old_hedged"] = float(O.q_hedged.mean())
ln["old_p_lt_0.8"] = float((O.p.astype(float) < 0.8).mean()) if O.p.notna().any() else None
res["label_noise"] = ln
# sample sentences (no ids) for manual audit of precision
smp = neg[neg.neg_frx_mention].sample(min(60, int(neg.neg_frx_mention.sum())), random_state=0)
smp[["cat", "neg_frx_sents"]].to_csv(OUT / "audit_noisy_neg_sentences_train.csv", index=False)

# ---- 6. errors at 90% specificity
thr = float(np.quantile(neg[S], 0.90)); res["thr_spec90"] = thr
for g in ["old", "acute", "not_stated"]:
    res[f"sens_spec90_{g}"] = float((T[T.grp == g][S] >= thr).mean())
FN = T[(T.grp == "old") & (T[S] < thr)].copy()


def fn_cat(r):
    if r["cat"] == "probably_yes" or r.q_hedged or (pd.notna(r.p) and float(r.p) < 0.8):
        return "標註不確定（語氣保留/probably_yes/p<0.8）"
    if r.q_hardware or r.q_cement:
        return "術後/金屬或骨水泥"
    if r.vert_out_fov or (r.site_rib and not r.site_vertebra and not r.site_pelv and pd.notna(r.rib_fov_frac) and r.rib_fov_frac == 0):
        return "超出 FOV/覆蓋範圍外（椎體>T8 以上、肋骨不在影像內）"
    if r.site_vertebra and r.vseg_missing:
        return "分割不完整（椎體<6 節）"
    if r.comp_yes and pd.notna(r.max_height_loss) and r.max_height_loss < 0.15:
        return "幾何細微（max_height_loss<0.15）"
    if r.site_rib and not r.site_vertebra:
        return "肋骨陳舊骨折（癒合/細微）"
    if r.site_other or r.sites == "":
        return "其他部位/未標部位"
    return "其他/無明顯原因"


FN["cat_err"] = FN.apply(fn_cat, axis=1)
t6a = FN.cat_err.value_counts().rename_axis("偽陰性類別").reset_index(name="n"); t6a["frac"] = t6a.n / len(FN)
t6a.to_csv(OUT / "t6a_old_fn_categories.csv", index=False)
res["n_old"] = int((T.grp == "old").sum()); res["n_old_fn"] = len(FN)
FP = neg[neg[S] >= thr].copy()
fpflags = {"age≥75（高齡）": FP.Age >= 75, "骨質疏鬆/減少": FP.r_osteopenia, "退化性變化": FP.r_degen, "Schmorl 結節": FP.r_schmorl,
           "脊椎側彎": FP.r_scolio, "金屬/術後": FP.r_hardware, "骨水泥": FP.r_cement, "報告提到陳舊骨折/壓迫變形（疑漏標）": FP.neg_frx_mention,
           "骨病灶/轉移": FP.r_lesion, "幾何 max_height_loss≥0.25": FP.max_height_loss >= 0.25}
allflags = {"age≥75（高齡）": neg.Age >= 75, "骨質疏鬆/減少": neg.r_osteopenia, "退化性變化": neg.r_degen, "Schmorl 結節": neg.r_schmorl,
            "脊椎側彎": neg.r_scolio, "金屬/術後": neg.r_hardware, "骨水泥": neg.r_cement, "報告提到陳舊骨折/壓迫變形（疑漏標）": neg.neg_frx_mention,
            "骨病灶/轉移": neg.r_lesion, "幾何 max_height_loss≥0.25": neg.max_height_loss >= 0.25}
t6b = pd.DataFrame([dict(flag=k, frac_in_FP=float(v.mean()), frac_in_all_neg=float(allflags[k].mean()), enrichment=float(v.mean() / max(allflags[k].mean(), 1e-9)))
                    for k, v in fpflags.items()])
t6b.to_csv(OUT / "t6b_fp_flags.csv", index=False)
res["n_fp"] = len(FP); res["fp_age_mean"] = float(FP.Age.mean())
res["fp_nothing_flagged"] = float((~pd.concat(fpflags, axis=1).fillna(False).any(axis=1)).mean())

# ---- 7. alternative evaluation definitions
ms_kw = q.str.contains(r"moderate|severe|marked|significant|burst|retropuls|plana|>\s?(?:40|50)", regex=True)
d["cs"] = (d.grp == "acute") | ((d.y == 1) & d.comp_yes & ms_kw) | ((d.y == 1) & d.site_hip_femur & ~d.q_hardware)
T = d[d.split == "train"].copy()
defs = {
    "目前定義：任何骨折 vs 陰性": T,
    "僅急性(含混合) vs 陰性（陳舊/未註明 排除）": T[T.grp.isin(["neg", "acute"])],
    "急性(含混合) vs 其他全部（陳舊視為陰性）": T.assign(y=(T.grp == "acute").astype(int)),
    "臨床顯著（急性 ∪ 中重度壓迫 ∪ 髖/股骨非術後） vs 陰性": T[(T.y == 0) | T.cs],
    "排除 單純陳舊肋骨 陽性": T[~((T.grp == "old") & T.site_rib & (T.n_sites == 1))],
    "排除 疑漏標陰性（報告提到陳舊骨折）": T[~T.neg_frx_mention],
}
rows = []
for nm, X in defs.items():
    rows.append(dict(definition=nm, n_pos=int(X.y.sum()), n_neg=int((X.y == 0).sum()), auc_model=auc(X.y, X[S]), auc_age=auc(X.y, X.Age),
                     auc_model_age_matched=maucs(X)))
t7 = pd.DataFrame(rows); t7.to_csv(OUT / "t7_definitions.csv", index=False)


# ---- FOV / unseeable share and simple ceiling arithmetic
O = T[T.grp == "old"]
uns = O.vert_out_fov | (O.site_rib & ~O.site_vertebra & ~O.site_pelv & (O.rib_fov_frac == 0))
res["old_unseeable_frac"] = float(uns.mean()); res["old_unseeable_n"] = int(uns.sum())
X = sub_vs_neg(T, (T.grp == "old") & uns); res["old_unseeable_auc"] = auc(X.y, X[S])
res["old_vert_out_fov_n"] = int(O.vert_out_fov.sum()); res["old_rib_fov0_n"] = int((O.site_rib & (O.rib_fov_frac == 0)).sum())
res["old_rib_fov_frac_mean"] = float(O[O.site_rib].rib_fov_frac.mean())
# AUROC decomposes as n_pos-weighted mean of subgroup AUROCs vs the same negatives
def cf(mask_pos, target):
    X = sub_vs_neg(T, mask_pos); a0 = auc(X.y, X[S]); w = mask_pos.sum()
    return dict(n=int(w), auc_now=a0, target=target, d_old=float(w / (T.grp == "old").sum() * (target - a0)), d_all=float(w / T.y.sum() * (target - a0)))
ribonly_old = (T.grp == "old") & T.site_rib & (T.n_sites == 1)
pelvonly_old = (T.grp == "old") & T.site_pelv & (T.n_sites == 1)
X = sub_vs_neg(T, (T.grp == "acute") & T.site_rib); rib_acute = auc(X.y, X[S])
res["cf_ribonly_old_to_acute_rib"] = cf(ribonly_old, rib_acute)
res["cf_pelvonly_old_to_0.837"] = cf(pelvonly_old, 0.837)
res["cf_unseeable_old_to_0.5_already"] = cf((T.grp == "old") & uns, 0.5)
a_, b_ = 0.054, 0.002
res["ceiling_label_noise"] = (1 - a_) * (1 - b_) + 0.5 * ((1 - a_) * b_ + a_ * (1 - b_))

# ---- val: descriptive only
rows = []
for g in ["all", "old", "acute", "not_stated"]:
    m = V.y == 1 if g == "all" else V.grp == g
    X = sub_vs_neg(V, m)
    rows.append(dict(group=GN[g], n_pos=int(X.y.sum()), auc_model=auc(X.y, X[S]), auc_age=auc(X.y, X.Age)))
tv = pd.DataFrame(rows); tv.to_csv(OUT / "tv_val_descriptive.csv", index=False)
res["val_noisy_neg_frac"] = float(V[V.y == 0].neg_frx_mention.mean())

json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=float, ensure_ascii=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for nm, t in [("t1", t1), ("t1b", t1b), ("t2", t2), ("t3", t3), ("t4", t4), ("t4b", t4b), ("t6a", t6a), ("t6b", t6b), ("t7", t7), ("val", tv)]:
    print("=====", nm); print(t.round(3).to_string(index=False))
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
