"""Tier B: refit the SELECTED region-unit stack of ru_stack.py / nu_stack.py exactly (same T frame, columns, candidate,
fill/log rules), check it reproduces the stored val scores (region_units/<unit>/stack/scores_val.csv), save it, and score
CG10k rows.  CG inputs: REF head columns + CG image features (same columns as integrated_best/models/<fd>_features.json,
derived with ib_stack.derive), unit heads (cg10k_best/bprobe/<code>/out/val_scores.csv, CG rows), optional HU scalars.
usage: bstack_apply.py FD UNIT SELECTED_NAME CODE CG_HEADS_CSV CG_FEAT_CSV [MERLIN_HUF CG_HUF]  -> cg10k_best/bstack/<fd>/"""
import sys, os, json, joblib, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); RU = H + "mw_exp/outputs_final/region_units/"; IBS = H + "mw_exp/outputs_final/integrated_best/"
FD, UNIT, SEL, CODE, CGH, CGF = sys.argv[1:7]; HUF = sys.argv[7] if len(sys.argv) > 7 else None; CGHUF = sys.argv[8] if len(sys.argv) > 8 else None
IB = RU + "ibroot/"; UD = RU + UNIT + "/"; OUT = H + f"cg10k_best/bstack/{FD}/"; os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], IB]; sys.path.insert(0, IBS + "scripts")
import ib_stack as ibs
from ib_core import CANDS, Prep, score, auc
c = ibs.CFG[FD]; A = IB + f"an/{c['an']}/"
lab = pd.read_csv(IB + "base/labels14_trval.csv")[["study_id", "split", FD]].rename(columns={FD: "y"})
oof = pd.read_csv(A + "oof_train.csv"); sv = pd.read_csv(A + "scores_val_descriptive.csv").drop(columns=["y_sp"], errors="ignore")
F, fcols = ibs.load_feats(c)
uo = pd.read_csv(UD + "out/oof_train.csv").drop(columns=["fold"]); uv = pd.read_csv(UD + "out/val_scores.csv")
UH0 = [x for x in uo.columns if x != "study_id"]; ren = {x: "ru_" + x for x in UH0}; UH = list(ren.values())
uo = uo.rename(columns=ren); uv = uv.rename(columns=ren)
T = lab[lab.split == "train"].merge(oof, on="study_id").merge(F, on="study_id", how="left").merge(uo, on="study_id", how="left")
T = T[T.y.notna()].reset_index(drop=True)
V = lab[lab.split == "val"].merge(sv, on="study_id").merge(F, on="study_id", how="left").merge(uv, on="study_id", how="left")
V = V[V.y.notna()].reset_index(drop=True)
# CG frame: heads + features (ib_apply rules) + unit heads (refit run, CG rows)
PRE = os.environ.get("CG_PREFIX", "")            # "D_" for the delay set (unit-head ids carry the prefix)
OUT = OUT.rstrip("/") + (f"_{PRE.rstrip('_').lower()}/" if PRE else "/"); os.makedirs(OUT, exist_ok=True)
G = pd.read_csv(CGF); G = G[G.err.fillna(0) == 0] if "err" in G else G
G = G[["study_id"] + [x for x in G.columns if x not in ibs.DROP and x not in ibs.META_COLS and pd.api.types.is_numeric_dtype(G[x])]]
G = ibs.derive(G, c)
cu = pd.read_csv(H + f"cg10k_best/bprobe/{CODE}/out/val_scores.csv").rename(columns=ren)
C = pd.read_csv(CGH).merge(G, on="study_id", how="left")
C["study_id"] = PRE + C.study_id.astype(str)
C = C.merge(cu, on="study_id", how="left")
HC = []
if HUF:
    hf = pd.read_csv(HUF).drop_duplicates("study_id"); HC = [x for x in hf.columns if x != "study_id"]
    T = T.merge(hf, on="study_id", how="left"); V = V.merge(hf, on="study_id", how="left")
    hc = pd.read_csv(CGHUF).drop_duplicates("study_id"); hc["study_id"] = PRE + hc.study_id.astype(str)
    C = C.merge(hc, on="study_id", how="left")
med = {}
for h in UH:
    med[h] = T[h].median(); T[h] = T[h].fillna(med[h]); V[h] = V[h].fillna(med[h]); C[h] = C[h].fillna(med[h])
for h in HC:
    lg = h.endswith(("_n300", "_n130"))
    for D in (T, V, C):
        D[h] = np.log1p(D[h].clip(lower=0)) if lg else D[h]
    md = T[h].median(); med[h] = md
    for D in (T, V, C): D[h] = D[h].fillna(md)
ref = json.load(open(IBS + f"models/{FD}_features.json")); RC = ref["columns"]
r = pd.read_csv(UD + "stack/results.csv"); row = r[r.model == SEL].iloc[0]; cn = row.cand
cols = {"REF": RC, "REF＋全部單元頭": RC + UH, "REF＋HU": RC + HC, "REF＋全部單元頭＋HU": RC + UH + HC}.get(SEL) or RC + ["ru_" + SEL.split("＋", 1)[1]]
y = T.y.values.astype(int)
pr = Prep().fit(T[cols]); m = CANDS[cn]().fit(pr.transform(T[cols]).values, y)
v = score(m, pr.transform(V[cols]).values)
sv_ = pd.read_csv(UD + "stack/scores_val.csv")[["study_id", SEL]].merge(pd.DataFrame({"study_id": V.study_id, "new": v}), on="study_id")
d = np.abs(sv_[SEL] - sv_.new).max()
print(f"{FD} {SEL} {cn}: val AUROC refit {auc(V.y.values.astype(int), v):.4f} stored {row.val:.4f} | max|d| vs stored {d:.2e}", flush=True)
miss = [x for x in cols if x not in C.columns]
assert not miss, miss
s = score(m, pr.transform(C[cols]).values)
joblib.dump(dict(finding=FD, model=m, prep=pr, cols=cols, cand=cn, selected=SEL, unit_fill=med), OUT + "stack.joblib")
pd.DataFrame({"study_id": C.study_id.str.replace("^D_", "", regex=True), FD: s}).to_csv(OUT + "cg_scores.csv", index=False)
json.dump(dict(finding=FD, selected=SEL, cand=cn, val_refit=float(auc(V.y.values.astype(int), v)), val_stored=float(row.val), max_abs_diff=float(d),
               n_cg=int(len(C)), cg_unit_head_missing=int(C[UH].isna().any(axis=1).sum())), open(OUT + "summary.json", "w"), indent=1)
