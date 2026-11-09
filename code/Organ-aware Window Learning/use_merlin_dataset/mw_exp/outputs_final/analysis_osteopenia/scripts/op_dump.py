"""Dump lumbar_spine + L1 unit embeddings for train/val ids only (no test ids in this matrix: *_notest.npz)."""
import numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
units = [str(u) for u in z["units"]]; gi = [units.index("lumbar_spine"), units.index("L1")]
ids = np.array([str(i) for i in z["ids"]]); E = z["E"][:, gi]; P = z["P"][:, gi]
lab = set(pd.read_csv(R + "analysis_osteopenia/labels_trval.csv").study_id)
k = np.array([i in lab for i in ids])
np.savez(R + "analysis_osteopenia/emb_lumbar_L1_trval.npz", E=E[k].astype(np.float16), P=P[k], ids=ids[k], units=np.array(["lumbar_spine", "L1"]))
print("saved", E[k].shape, E.dtype)
