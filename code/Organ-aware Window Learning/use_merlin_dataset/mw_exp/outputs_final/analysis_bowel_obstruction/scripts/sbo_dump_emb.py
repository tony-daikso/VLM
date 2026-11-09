"""Dump train/val-only embedding slice (small_bowel, colon, stomach units) for the bowel-obstruction OOF.
Test ids are never in the source matrix (notest) and are additionally filtered by split.csv."""
import numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_bowel_obstruction/"
spl = pd.read_csv("/home/advantech/Desktop/Tony/VLM/OWL/data/split.csv").set_index("study_id")["split"]
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = np.array(z["ids"]); units = list(z["units"]); U = ["small_bowel", "colon", "stomach"]; UI = [units.index(u) for u in U]
keep = np.array([spl.get(s) in ("train", "val") for s in ids])
E = z["E"][:, UI][keep]; P = z["P"][:, UI][keep]
np.savez(A + "emb_sb_co_st_trval.npz", E=E, P=P, ids=ids[keep], units=np.array(U))
print(E.shape, E.dtype, P.shape, int(keep.sum()), "dropped", int((~keep).sum()))
