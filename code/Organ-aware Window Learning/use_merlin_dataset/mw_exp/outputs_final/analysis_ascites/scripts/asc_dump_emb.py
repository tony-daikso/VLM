"""Dump the 12 main-unit embeddings of TRAIN+VAL rows only (from the _notest matrix) to a float16 npz for the OOF host."""
import numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = z["ids"]; spl = pd.read_csv("/home/advantech/Desktop/Tony/VLM/OWL/data/split.csv").set_index("study_id")["split"]
k = np.array([spl.get(s) in ("train", "val") for s in ids])
E = z["E"]; print(E.dtype, E.shape)
np.savez(R + "analysis_ascites/emb12_trval.npz", E=E[k][:, :12].astype(np.float16), P=z["P"][k][:, :12], ids=ids[k], units=z["units"][:12])
print("saved", k.sum())
