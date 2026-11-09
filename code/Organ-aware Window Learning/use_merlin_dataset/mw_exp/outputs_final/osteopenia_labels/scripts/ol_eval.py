"""Evaluate osteopenia label-handling schemes on val under two label sets. Train/val only; test never read.
  (1) orig : report labels, all val
  (2) clean: val with report-negative & L1 HU < 110 removed (circular w.r.t. HU, reported as such)
usage: python ol_eval.py OUT_DIR"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ol_common import D, load_base
sys.path.insert(0, D)
from ib_core import auc, age_matched_auc, spearman_neg, sens_at_spec, boot_ci, boot_delta, EDGES

OUT = sys.argv[1].rstrip("/") + "/"
B = load_base(); T, V = B["T"], B["V"]
tr_ids, va_ids = list(T.index), list(V.index)
SCH = ["A", "B110", "B130", "C", "D", "Dmt"]
NAMES = {"A": "A 原本（報告陽性 vs 其他）", "B110": "B 排除 HU<110 報告陰性", "B130": "B130 排除 HU<130 報告陰性",
         "C": "C PU learning（Elkan-Noto）", "D": "D soft label max(報告, σ((110−HU)/10))",
         "Dmt": "D2 多任務堆疊（＋HU<110 head）", "E": "E 只用 −L1 HU（不訓練）"}
oof, val = {}, {}
for s in SCH:
    if not os.path.exists(OUT + f"{s}_val.csv"): continue
    oof[s] = pd.read_csv(OUT + f"{s}_oof.csv").set_index("study_id").loc[tr_ids, "stack"].values
    val[s] = pd.read_csv(OUT + f"{s}_val.csv").set_index("study_id").loc[va_ids, "stack"].values
med_hu = np.nanmedian(T.hu.values)
oof["E"] = -np.nan_to_num(T.hu.values, nan=med_hu); val["E"] = -np.nan_to_num(V.hu.values, nan=med_hu)
# formal best from osteopenia_v15 (reproduction check of A)
fv = pd.read_csv(D + "stack_val.csv").set_index("study_id")["5win+v15"].reindex(va_ids).values
fo = pd.read_csv(D + "stack_oof_train.csv").set_index("study_id")["5win+v15"].reindex(tr_ids).values
models = [s for s in SCH if s in val] + ["E"]

def hu_matched_auc(y, s, hu):
    b = pd.cut(pd.Series(hu), [-1e9, 110, 130, 160, 200, 1e9], right=False).values; num = den = 0.0
    for g in pd.unique(b[~pd.isna(b)]):
        m = np.asarray(b == g); n1 = y[m].sum(); n0 = m.sum() - n1
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den if den else np.nan

def strat_auc(y, s, age, sex):
    b = pd.cut(pd.Series(age), EDGES, right=False).astype(str).values
    key = np.array([f"{a}|{x}" for a, x in zip(b, pd.Series(sex).fillna("NA").astype(str))]); num = den = 0.0
    for k in np.unique(key):
        m = key == k; n1 = y[m].sum(); n0 = m.sum() - n1
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den

yV = V.y.values; huV = V.hu.values; ageV = V.age.values; sexV = V.sex.values
sets = {"orig": np.ones(len(V), bool), "clean110": ~((yV == 0) & (huV < 110))}
rows = []
for lsn, m in sets.items():
    y, age, sex, hu = yV[m], ageV[m], sexV[m], huV[m]; ref = val["A"][m]
    for s in models:
        v = val[s][m]
        r = dict(label_set=lsn, scheme=s, name=NAMES[s], n=int(m.sum()), n_pos=int(y.sum()),
                 auc=auc(y, v), ci=boot_ci(y, v))
        if s != "A":
            d = boot_delta(y, ref, v); r.update(d=d["mean"], d_ci=d["ci"])
            da = boot_delta(y, ref, v, n=500, age=age); r.update(d_age_ci=da["ci"])
        r["age_matched"] = age_matched_auc(y, v, age); r["agesex"] = strat_auc(y, v, age, sex)
        hm = ~np.isnan(hu); r["hu_matched"] = hu_matched_auc(y[hm], v[hm], hu[hm])
        r["spearman_neg"] = spearman_neg(y, v, age); r["sens90"] = sens_at_spec(y, v)[0]
        rows.append(r); print(lsn, s, f"{r['auc']:.4f}", r.get("d"), f"age {r['age_matched']:.4f} hu-m {r['hu_matched']:.4f}", flush=True)
    r = dict(label_set=lsn, scheme="formal_v15", auc=auc(y, fv[m])); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(OUT + "val_results.csv", index=False)

# train OOF (orig and cleaned train labels)
yT = T.y.values; cT = ~((yT == 0) & (T.hu.values < 110))
tro = [dict(scheme=s, oof_orig=auc(yT, oof[s]), oof_clean110=auc(yT[cT], oof[s][cT]),
            oof_age_orig=age_matched_auc(yT, oof[s], T.age.values)) for s in models]
tro.append(dict(scheme="formal_v15", oof_orig=auc(yT, fo), oof_clean110=auc(yT[cT], fo[cT])))
pd.DataFrame(tro).to_csv(OUT + "train_oof.csv", index=False)

# HU<110 binary rule
rule = []
for lsn, m in sets.items():
    y, hu, age = yV[m], huV[m], ageV[m]
    for thr in (110, 130, 160):
        p = hu < thr
        for sub, sm in (("all", np.ones(len(y), bool)), (">=60y", age >= 60)):
            yy, pp = y[sm], p[sm]
            rule.append(dict(label_set=lsn, thr=thr, subset=sub, sens=pp[yy == 1].mean(), spec=1 - pp[yy == 0].mean(),
                             ppv=yy[pp].mean() if pp.sum() else np.nan, n_flag=int(pp.sum())))
    # model at same specificity as the HU<110 rule
    spec110 = 1 - (hu < 110)[y == 0].mean()
    for s in models:
        se, _ = sens_at_spec(y, val[s][m], spec=spec110)
        rule.append(dict(label_set=lsn, thr="model@spec(HU<110)", subset=s, sens=se, spec=spec110))
pd.DataFrame(rule).to_csv(OUT + "hu_rule.csv", index=False)

# relabel counts + age distribution
yT = T.y.values
rel = []
for nm, X in (("train", T), ("val", V)):
    for thr in (110, 130):
        mm = (X.y == 0) & (X.hu < thr); a = X.age[mm]; an = X.age[(X.y == 0) & ~(X.hu < thr)]; ap = X.age[X.y == 1]
        rel.append(dict(split=nm, rule=f"報告陰性且 HU<{thr}", n=int(mm.sum()), pct_neg=mm.sum() / (X.y == 0).sum(),
                        ratio_to_pos=mm.sum() / X.y.sum(), age_med=a.median(), age_q1=a.quantile(.25), age_q3=a.quantile(.75),
                        pct_ge70=(a >= 70).mean(), pct_ge80=(a >= 80).mean(), pct_lt60=(a < 60).mean(),
                        pct_female=(X.sex[mm] == "Female").mean(), age_med_other_neg=an.median(), age_med_pos=ap.median(),
                        pct_frac=(X.frac[mm] == 1).mean()))
    if nm == "train" and os.path.exists(OUT + "A_oof.csv"):
        from sklearn.linear_model import LogisticRegression
        from ol_common import scheme_targets
        a = pd.read_csv(OUT + "A_oof.csv").set_index("study_id").loc[tr_ids, "stack"].values
        g = LogisticRegression(C=1e6, max_iter=1000).fit(a[:, None], yT).predict_proba(a[:, None])[:, 1]
        c = g[yT == 1].mean(); t, _ = scheme_targets(T, "C", pu_g=(g, c))
        for lo in (0.5, 0.2):
            mm = (X.y == 0) & (t > lo); a_ = X.age[mm]
            rel.append(dict(split=nm, rule=f"C：PU 權重 p>{lo} 的報告陰性", n=int(mm.sum()), pct_neg=mm.sum() / (X.y == 0).sum(),
                            ratio_to_pos=mm.sum() / X.y.sum(), age_med=a_.median(), age_q1=a_.quantile(.25), age_q3=a_.quantile(.75),
                            pct_ge70=(a_ >= 70).mean(), pct_ge80=(a_ >= 80).mean(), pct_lt60=(a_ < 60).mean(),
                            pct_female=(X.sex[mm] == "Female").mean(), pct_frac=(X.frac[mm] == 1).mean(),
                            hu_med=X.hu[mm].median(), pct_hu_lt110=(X.hu[mm] < 110).mean()))
pd.DataFrame(rel).to_csv(OUT + "relabel.csv", index=False)
# per-decade table
dec = []
for nm, X in (("train", T), ("val", V)):
    b = pd.cut(X.age, [0, 50, 60, 70, 80, 200], right=False)
    for g in b.cat.categories:
        mg = b == g; neg = mg & (X.y == 0)
        dec.append(dict(split=nm, age=str(g), n_neg=int(neg.sum()), n_pos=int((mg & (X.y == 1)).sum()),
                        neg_lt110=int((neg & (X.hu < 110)).sum()), neg_lt130=int((neg & (X.hu < 130)).sum())))
pd.DataFrame(dec).to_csv(OUT + "relabel_by_age.csv", index=False)
# C / D implied relabels on train
ex = {}
if os.path.exists(OUT + "C_info.json"): ex["C"] = json.load(open(OUT + "C_info.json"))
ex["D_soft_gt_0.5_neg"] = int(((T.y == 0) & (T.hu < 110)).sum())
hu = T.hu.values; sd = 1 / (1 + np.exp(-(110 - hu) / 10)); sd = np.where(np.isnan(hu), 0, sd)
ex["D_sum_soft_neg"] = float(sd[T.y.values == 0].sum()); ex["D_neg_soft_gt_0.1"] = int(((T.y.values == 0) & (sd > 0.1)).sum())
for s in SCH:
    if os.path.exists(OUT + f"{s}_info.json"): ex[f"{s}_coef"] = json.load(open(OUT + f"{s}_info.json")).get("stack_coef")
json.dump(ex, open(OUT + "extra.json", "w"), indent=1, ensure_ascii=False)
print("EVAL_DONE")
