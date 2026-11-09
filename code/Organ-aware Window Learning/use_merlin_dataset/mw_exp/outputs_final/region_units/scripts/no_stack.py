"""Exp O score-level stack: CURRENT best (REF = per-case train OOF + val scores of the current best model of the finding) +
Exp O unit head scores (train OOF from no_probe.py, val = full-train refit). Level-2 = LR (C 0.1 / 1) or GBM on
[logit(REF) | heads], 5-fold train CV on the probe folds -> model chosen on train OOF only -> full-train refit -> val once.
Metrics (ib_core): AUROC + bootstrap CI, paired bootstrap delta vs REF (train OOF and val), age-matched AUROC (10-y bins) and
its paired delta, Spearman(score, age) on negatives, sensitivity at 90 % specificity; subgroup AUROC (subgroup positives vs
all negatives) with paired delta.  Age/sex never inputs.  Test never read (all inputs are train/val files).
Usage (in outputs_final/region_units): no_stack.py UNIT_DIR      writes UNIT_DIR/stack/{results.csv,results.json,scores_*.csv}"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "../integrated_best/scripts")
from ib_core import CANDS, Prep, score, auc, boot_ci, boot_delta, age_matched_auc, spearman_neg, sens_at_spec
UD = sys.argv[1].rstrip("/")
IB = "../integrated_best/scores/"
def lab_csv(c): return f"probe_inputs/labels_{c}.csv"
BD_REF = "cat4+E3_late+E0+porta_emb+HU+tube+porta|gbm"
def fa_sub(V):
    L = pd.read_csv("../analysis_free_air/labels_trval.csv")[["study_id", "fa_postop"]]
    s = V[["study_id"]].merge(L, on="study_id", how="left")
    return {"非術後陽性（含未註明）": (s.fa_postop != "yes").values, "術後陽性": (s.fa_postop == "yes").values}
def frp_sub(V):
    s = V[["study_id"]].merge(pd.read_csv("probe_inputs/frac_sites_trval.csv")[["study_id", "pelvis"]], on="study_id", how="left")
    return {"骨盆/薦椎/髖": (s.pelvis == 1).values, "其他部位": (s.pelvis != 1).values}
def go_sub(V):
    L = pd.read_csv("../analysis_bowel_obstruction/labels_trval.csv")[["study_id", "kw_gastric_outlet"]]
    s = V[["study_id"]].merge(L, on="study_id", how="left")
    return {"胃出口／十二指腸／輸入環": (s.kw_gastric_outlet == 1).values}
CFG = {
    "bile_duct_r2": dict(lab="bd", ref=("../biliary_improve/stack_bd5/stack_oof_train.csv", "../biliary_improve/stack_bd5/stack_val.csv", BD_REF, BD_REF),
                         outs=["out"], sub=None),
    "free_air_r2": dict(lab="fa2", ref=("free_air/stack/scores_oof.csv", "free_air/stack/scores_val.csv", "REF＋全部單元頭", "REF＋全部單元頭"),
                        outs=["out"], sub=fa_sub),
    "ascites_r2": dict(lab="asc2", ref=("ascites/stack/scores_oof.csv", "ascites/stack/scores_val.csv", "REF＋U_org", "REF＋U_org"), outs=["out"], sub=None),
    "fracture_pelvis_r2": dict(lab="frp", ref=(IB + "fracture_oof.csv", IB + "fracture_val.csv", "stack_oof", "stack"), outs=["out", "out_pel"], sub=frp_sub),
    "bowel_obstruction_r2": dict(lab="go", ref=(IB + "bowel_obstruction_oof.csv", IB + "bowel_obstruction_val.csv", "stack_oof", "stack"), outs=["out"], sub=go_sub),
}
c = CFG[UD]
lab = pd.read_csv(lab_csv(c["lab"])); assert set(lab.split) <= {"train", "val"}; lab = lab[lab.y.notna()]
age = pd.read_csv("ibroot/base/age_trval.csv")
ro = pd.read_csv(c["ref"][0])[["study_id", c["ref"][2]]].rename(columns={c["ref"][2]: "REF"})
rv = pd.read_csv(c["ref"][1])[["study_id", c["ref"][3]]].rename(columns={c["ref"][3]: "REF"})
HO, HV = None, None
for o in c["outs"]:
    a = pd.read_csv(f"{UD}/{o}/oof_train.csv").drop(columns=["fold"]); b = pd.read_csv(f"{UD}/{o}/val_scores.csv")
    sfx = "" if o == "out" else "_" + o.split("_", 1)[1]
    a = a.rename(columns={x: x + sfx for x in a.columns if x != "study_id"}); b = b.rename(columns={x: x + sfx for x in b.columns if x != "study_id"})
    HO = a if HO is None else HO.merge(a, on="study_id", how="outer"); HV = b if HV is None else HV.merge(b, on="study_id", how="outer")
UH = [x for x in HO.columns if x != "study_id"]
T = lab[lab.split == "train"].merge(ro, on="study_id").merge(HO, on="study_id", how="left").merge(age, on="study_id", how="left").reset_index(drop=True)
V = lab[lab.split == "val"].merge(rv, on="study_id").merge(HV, on="study_id", how="left").merge(age, on="study_id", how="left").reset_index(drop=True)
for D in (T, V):
    if D.REF.between(0, 1).all():
        D["REF"] = np.log(np.clip(D.REF, 1e-6, 1 - 1e-6) / np.clip(1 - D.REF, 1e-6, 1))
cov = (T[UH].notna().all(1).mean(), V[UH].notna().all(1).mean())
for h in UH:
    md = T[h].median(); T[h] = T[h].fillna(md); V[h] = V[h].fillna(md)
y, yv, folds = T.y.values.astype(int), V.y.values.astype(int), T.fold.values.astype(int)
print(f"[{UD}] train {len(T)} ({y.sum()} pos) val {len(V)} ({yv.sum()} pos); head coverage train {cov[0]:.4f} val {cov[1]:.4f}; REF OOF {auc(y, T.REF):.4f} val {auc(yv, V.REF):.4f}", flush=True)
CN = ["LR_C0.1", "LR_C1", "GBM_lr.03_l7_m40_500"]
def fit_eval(cols, cn):
    pr = Prep().fit(T[cols]); XT = pr.transform(T[cols]).values; o = np.zeros(len(T))
    for k in np.unique(folds):
        a, b = folds != k, folds == k
        o[b] = score(CANDS[cn]().fit(XT[a], y[a]), XT[b])
    return o, score(CANDS[cn]().fit(XT, y), pr.transform(V[cols]).values)
S = {"REF": (auc(y, T.REF), "-", T.REF.values, V.REF.values)}
sets = {}
for h in UH:
    S[f"{h} 單獨"] = (auc(y, T[h]), "probe", T[h].values, V[h].values)
    sets[f"REF＋{h}"] = ["REF", h]
if len(UH) > 1:
    sets["REF＋全部單元頭"] = ["REF"] + UH
for nm, cols in sets.items():
    best = None
    for cn in CN:
        o, v = fit_eval(cols, cn); a = auc(y, o)
        print(f"  {nm:<30s} {cn:<24s} OOF {a:.4f} val {auc(yv, v):.4f}", flush=True)
        if best is None or a > best[0]: best = (a, cn, o, v)
    S[nm] = best
subT = c["sub"](T) if c["sub"] else {}; subV = c["sub"](V) if c["sub"] else {}
ro_, rv_ = S["REF"][2], S["REF"][3]
res = []
for nm, (a, cn, o, v) in S.items():
    r = dict(model=nm, cand=cn, n_tr=len(y), pos_tr=int(y.sum()), n_val=len(yv), pos_val=int(yv.sum()), oof=a, val=auc(yv, v),
             val_ci=boot_ci(yv, v), oof_age=age_matched_auc(y, o, T.Age.values), val_age=age_matched_auc(yv, v, V.Age.values),
             spearman_neg_val=spearman_neg(yv, v, V.Age.values), sens90_val=sens_at_spec(yv, v)[0],
             sens90_val_oofthr=sens_at_spec(yv, v, sens_at_spec(y, o)[1])[0])
    if nm != "REF":
        r["d_oof"] = boot_delta(y, ro_, o); r["d_val"] = boot_delta(yv, rv_, v); r["d_val_age"] = boot_delta(yv, rv_, v, n=500, age=V.Age.values)
    for sn in subV:
        mt = (y == 0) | ((y == 1) & subT[sn]); mv = (yv == 0) | ((yv == 1) & subV[sn])
        r[f"sub_{sn}_n_val"] = int(((yv == 1) & subV[sn]).sum()); r[f"sub_{sn}_oof"] = auc(y[mt], o[mt]); r[f"sub_{sn}_val"] = auc(yv[mv], v[mv])
        if nm != "REF":
            r[f"sub_{sn}_d_val"] = boot_delta(yv[mv], rv_[mv], v[mv])
    res.append(r)
    print(f"{nm:<30s} OOF {r['oof']:.4f} val {r['val']:.4f} {np.round(r['val_ci'], 3)} age {r['val_age']:.4f} sp {r['spearman_neg_val']:+.3f}"
          + (f" dVal {r['d_val']['mean']:+.4f} {np.round(r['d_val']['ci'], 4)} dOOF {r['d_oof']['mean']:+.4f} {np.round(r['d_oof']['ci'], 4)} dAge {r['d_val_age']['mean']:+.4f} {np.round(r['d_val_age']['ci'], 4)}" if nm != "REF" else "")
          + "".join(f" | {sn} val {r[f'sub_{sn}_val']:.4f}" for sn in subV), flush=True)
sel = max([r for r in res if r["model"].startswith("REF＋")], key=lambda r: r["oof"])
print(f"SELECTED (train OOF): {sel['model']} ({sel['cand']}) OOF {sel['oof']:.4f} val {sel['val']:.4f}", flush=True)
os.makedirs(UD + "/stack", exist_ok=True)
flat = []
for r in res:
    f = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
    f["val_lo"], f["val_hi"] = r["val_ci"]
    for k in [k for k in r if k.startswith("d_") or k.endswith("_d_val")]:
        f[k], f[k + "_lo"], f[k + "_hi"] = r[k]["mean"], *r[k]["ci"]
    f["selected"] = r is sel; flat.append(f)
pd.DataFrame(flat).to_csv(UD + "/stack/results.csv", index=False)
json.dump(dict(results=res, selected=sel["model"], ref_source=c["ref"]), open(UD + "/stack/results.json", "w"), indent=1, default=float)
pd.DataFrame({"study_id": T.study_id, "fold": folds, "y": y, **{k: v[2] for k, v in S.items()}}).to_csv(UD + "/stack/scores_oof.csv", index=False)
pd.DataFrame({"study_id": V.study_id, "y": yv, **{k: v[3] for k, v in S.items()}}).to_csv(UD + "/stack/scores_val.csv", index=False)
print("RU_STACK_DONE", flush=True)
