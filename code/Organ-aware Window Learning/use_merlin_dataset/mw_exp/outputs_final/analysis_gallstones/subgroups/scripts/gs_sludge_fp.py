"""Sludge-only (labelled negative) cases: V4 score distribution vs other negatives (train OOF / val). Train+val only."""
import pandas as pd, numpy as np
I = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
F = pd.read_csv(I + "analysis_gallstones/subgroups/case_flags_trval.csv")[["study_id", "sludge", "img_axis"]]
tr = pd.read_csv(I + "gallstones_improve/gs1/gs1_oof_train.csv").merge(F, on="study_id")
va = pd.read_csv(I + "gallstones_improve/gs1/gs1_val_scores.csv").merge(F, on="study_id")
for m in ["E3_late", "quickwins", "V4"]:
    thr = np.quantile(tr.loc[tr.y == 0, m], .9)
    for nm, D in [("train_oof", tr), ("val", va)]:
        n = D[D.y == 0]
        for g, x in n.groupby("sludge"):
            print(m, nm, g, "n_neg", len(x), "FP%@spec90", round(100 * (x[m] > thr).mean(), 1))
        for g, x in n.groupby("img_axis"):
            print(m, nm, "neg", g, len(x), "FP%", round(100 * (x[m] > thr).mean(), 1))
