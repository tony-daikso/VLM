"""GS1 ablation (train OOF only): drop feature groups from V4 with the V4-selected GBM."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], "in", "out/gs1_feat3mm.csv", "out/gs1_abl"]
src = open("scripts/gs1_stack.py").read().split("VARIANTS = {")[0]
exec(src)
from sklearn.metrics import roc_auc_score
mk = CANDS["GBM_lr.05_l31_m80_200"]
def run(drop, label):
    s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, med = design(T[tr], ["hu", "ag", "ch", "g1"]); Xte, _ = design(T[te], ["hu", "ag", "ch", "g1"], med)
        cols = [c for c in Xtr.columns if not any(c.startswith(p) for p in drop)]
        s[te] = score(mk().fit(Xtr[cols], y[tr]), Xte[cols])
    k = (y == 0) | ((T.gb_maxblob100.fillna(0) < 2) & (T.gb_vox.fillna(0) >= 100)).values
    print(label, round(roc_auc_score(y, s), 4), "nonvis", round(roc_auc_score(y[k], s[k]), 4), "age-matched", round(age_matched_auc(y, s, T.Age), 4), flush=True)
run([], "V4 full")
run(["ag_liver_vox"], "-liver_vox")
run(["ag_gb_frac_"], "-GB position in liver bbox")
run(["ag_f30", "ag_f50", "ag_peri", "ag_body_metal", "gb_missing_x_clip"], "-fossa/clip")
run(["ag_gb_liver_contact", "ag_gb_bowel_contact", "ag_gb_ncc", "ag_gb_lcc_frac"], "-GB contact/shape")
run(["chole"], "-chole head")
run(["g1_"], "-3mm texture")
