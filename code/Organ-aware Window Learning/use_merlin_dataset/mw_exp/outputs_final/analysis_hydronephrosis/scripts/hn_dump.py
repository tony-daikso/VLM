"""Dump kidney-unit embeddings (train+val ids only; test never written) for local OOF. Also the 12-unit train mean/std
used by probe.py --standardize is NOT needed: we standardize on train kidney rows like the templates."""
import numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = np.array(z["ids"]); units = list(z["units"]); ku = [units.index("kidney_left"), units.index("kidney_right")]
lab = pd.read_csv(R + "analysis_hydronephrosis/labels_trval.csv")
keep = np.where(np.isin(ids, lab.study_id.values))[0]
E = z["E"][keep][:, ku].astype(np.float16); P = z["P"][keep][:, ku]
np.savez(R + "analysis_hydronephrosis/kidney_emb_trval.npz", ids=ids[keep], E=E, P=P, units=np.array(["kidney_left", "kidney_right"]))
print(E.shape, P.shape, P.mean(0), [k for k in z.files])
