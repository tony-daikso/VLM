"""Exp N (copy of Exp M ru_stack.py; optional 4th arg = per-crop HU scalar csv from nu_hufeat.py, added as
extra sets REF＋HU and REF＋全部單元頭＋HU). Exp M: stack region-unit probe scores onto the integrated_best REF stack of a finding (train 5-fold OOF selection ->
full-train refit -> val once). REF = integrated_best models/<finding>_features.json columns + chosen candidate, rebuilt
here with ib_stack's own loaders (checked against integrated_best/scores/<finding>_oof.csv / _val.csv).
New feature sets: REF + each unit head, REF + all unit heads; candidates: REF's own + LR_C0.1 + GBM_lr.03_l7_m40_500.
Metrics (ib_core): AUROC + bootstrap CI, paired bootstrap delta vs REF (train OOF and val), age-matched AUROC (10-y bins)
and its delta vs REF, Spearman(score, age) on negatives, sensitivity at 90 % specificity. Age/sex never inputs.
Usage: ru_stack.py IB_ROOT FINDING UNIT_DIR    (UNIT_DIR/out/{oof_train,val_scores}.csv from ru_probe.py)
Writes UNIT_DIR/stack/{results.csv, results.json, scores_oof.csv, scores_val.csv}."""
import sys, os, json, numpy as np, pandas as pd
IB, FD, UD = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3].rstrip("/") + "/"
HUF = sys.argv[4] if len(sys.argv) > 4 else None
IBS = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best/"
sys.argv = [sys.argv[0], IB]
sys.path.insert(0, IBS + "scripts")
import ib_stack as ibs
from ib_core import CANDS, Prep, score, auc, boot_ci, boot_delta, age_matched_auc, spearman_neg, sens_at_spec

c = ibs.CFG[FD]; A = IB + f"an/{c['an']}/"
lab = pd.read_csv(IB + "base/labels14_trval.csv")[["study_id", "split", FD]].rename(columns={FD: "y"})
assert set(lab.split) <= {"train", "val"}
age = pd.read_csv(IB + "base/age_trval.csv")
exb = pd.read_csv(IB + "base/expB_E3late_val.csv")[["study_id", FD]].rename(columns={FD: "expB_E3"})
oof = pd.read_csv(A + "oof_train.csv"); sv = pd.read_csv(A + "scores_val_descriptive.csv").drop(columns=["y_sp"], errors="ignore")
F, fcols = ibs.load_feats(c)
uo = pd.read_csv(UD + "out/oof_train.csv").drop(columns=["fold"]); uv = pd.read_csv(UD + "out/val_scores.csv")
UH = [x for x in uo.columns if x != "study_id"]
uo = uo.rename(columns={x: "ru_" + x for x in UH}); uv = uv.rename(columns={x: "ru_" + x for x in UH}); UH = ["ru_" + x for x in UH]
T = lab[lab.split == "train"].merge(oof, on="study_id").merge(F, on="study_id", how="left").merge(age, on="study_id", how="left").merge(uo, on="study_id", how="left")
HC = []
if HUF:
    hf = pd.read_csv(HUF).drop_duplicates("study_id"); HC = [c for c in hf.columns if c != "study_id"]
    T = T.merge(hf, on="study_id", how="left")
T = T[T.y.notna()].reset_index(drop=True)
V = lab[lab.split == "val"].merge(sv, on="study_id").merge(exb, on="study_id").merge(F, on="study_id", how="left").merge(age, on="study_id", how="left").merge(uv, on="study_id", how="left")
if HUF:
    V = V.merge(hf, on="study_id", how="left")
V = V[V.y.notna()].reset_index(drop=True)
for h in HC:
    T[h] = np.log1p(T[h].clip(lower=0)) if h.endswith(("_n300", "_n130")) else T[h]
    V[h] = np.log1p(V[h].clip(lower=0)) if h.endswith(("_n300", "_n130")) else V[h]
    md = T[h].median(); T[h] = T[h].fillna(md); V[h] = V[h].fillna(md)
y = T.y.values.astype(int); yv = V.y.values.astype(int); folds = T.fold.values
print(f"[{FD}] train {len(T)} ({y.sum()} pos) val {len(V)} ({yv.sum()} pos); unit-head coverage train {T[UH].notna().all(1).mean():.4f} val {V[UH].notna().all(1).mean():.4f}", flush=True)
for h in UH:
    T[h] = T[h].fillna(T[h].median()); V[h] = V[h].fillna(T[h].median())
ref = json.load(open(IBS + f"models/{FD}_features.json")); RC, rcand = ref["columns"], ref["cand"]


def fit_eval(cols, cn):
    pr = Prep().fit(T[cols]); XT = pr.transform(T[cols]).values; o = np.zeros(len(T))
    for k in np.unique(folds):
        a, b = folds != k, folds == k
        o[b] = score(CANDS[cn]().fit(XT[a], y[a]), XT[b])
    m = CANDS[cn]().fit(XT, y)
    return o, score(m, pr.transform(V[cols]).values)


R0 = fit_eval(RC, rcand)
chk_o = pd.read_csv(IBS + f"scores/{FD}_oof.csv"); chk_v = pd.read_csv(IBS + f"scores/{FD}_val.csv")
print(f"REF rebuilt: OOF {auc(y, R0[0]):.4f} val {auc(yv, R0[1]):.4f} | integrated_best files: OOF {auc(chk_o.y, chk_o.stack_oof):.4f} val {auc(chk_v["y"], chk_v["stack"]):.4f}", flush=True)
sets = {"REF": (RC, [rcand])}
for h in UH:
    sets[f"{h[3:]} 單獨"] = ([h], ["LR_C0.1"])
    sets[f"REF＋{h[3:]}"] = (RC + [h], sorted({rcand, "LR_C0.1"}))
if len(UH) > 1:
    sets["REF＋全部單元頭"] = (RC + UH, sorted({rcand, "LR_C0.1"}))
if HC:
    sets["REF＋HU"] = (RC + HC, sorted({rcand, "LR_C0.1"}))
    sets["REF＋全部單元頭＋HU"] = (RC + UH + HC, sorted({rcand, "LR_C0.1"}))
res, S = [], {}
for nm, (cols, cands) in sets.items():
    best = None
    for cn in cands:
        o, v = (R0 if nm == "REF" else fit_eval(cols, cn))
        a = auc(y, o); print(f"  {nm:<28s} {cn:<24s} OOF {a:.4f} val {auc(yv, v):.4f}", flush=True)
        if best is None or a > best[0]: best = (a, cn, o, v)
    S[nm] = best
ro, rv = S["REF"][2], S["REF"][3]
for nm, (a, cn, o, v) in S.items():
    r = dict(finding=FD, model=nm, cand=cn, n_tr=len(y), pos_tr=int(y.sum()), n_val=len(yv), pos_val=int(yv.sum()),
             oof=a, val=auc(yv, v), val_ci=boot_ci(yv, v), oof_age=age_matched_auc(y, o, T.Age.values),
             val_age=age_matched_auc(yv, v, V.Age.values), spearman_neg_val=spearman_neg(yv, v, V.Age.values),
             sens90_val=sens_at_spec(yv, v)[0], sens90_val_oofthr=sens_at_spec(yv, v, sens_at_spec(y, o)[1])[0])
    if nm != "REF":
        r["d_oof"] = boot_delta(y, ro, o); r["d_val"] = boot_delta(yv, rv, v)
        r["d_val_age"] = boot_delta(yv, rv, v, n=500, age=V.Age.values)
    res.append(r)
    print(f"{nm:<28s} OOF {r['oof']:.4f} val {r['val']:.4f} {np.round(r['val_ci'], 3)} age {r['val_age']:.4f} sp {r['spearman_neg_val']:+.3f}"
          + (f" dVal {r['d_val']['mean']:+.4f} {np.round(r['d_val']['ci'], 4)} dOOF {r['d_oof']['mean']:+.4f} {np.round(r['d_oof']['ci'], 4)}" if nm != "REF" else ""), flush=True)
cand_new = [r for r in res if r["model"].startswith("REF＋")]
sel = max(cand_new, key=lambda r: r["oof"])
print(f"SELECTED (train OOF): {sel['model']} OOF {sel['oof']:.4f} val {sel['val']:.4f}", flush=True)
os.makedirs(UD + "stack", exist_ok=True)
flat = []
for r in res:
    f = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
    f["val_lo"], f["val_hi"] = r["val_ci"]
    for k in ("d_oof", "d_val", "d_val_age"):
        if k in r: f[k], f[k + "_lo"], f[k + "_hi"] = r[k]["mean"], *r[k]["ci"]
    f["selected"] = r is sel; flat.append(f)
pd.DataFrame(flat).to_csv(UD + "stack/results.csv", index=False)
json.dump(dict(results=res, selected=sel["model"]), open(UD + "stack/results.json", "w"), indent=1, default=float)
pd.DataFrame({"study_id": T.study_id, "fold": folds, "y": y, **{k: v[2] for k, v in S.items()}}).to_csv(UD + "stack/scores_oof.csv", index=False)
pd.DataFrame({"study_id": V.study_id, "y": yv, **{k: v[3] for k, v in S.items()}}).to_csv(UD + "stack/scores_val.csv", index=False)
print("RU_STACK_DONE", flush=True)
