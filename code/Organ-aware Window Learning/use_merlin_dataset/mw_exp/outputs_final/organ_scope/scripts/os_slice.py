"""organ_scope step 0: train+val-only slices of the cached embedding matrix (the _notest matrix has no test rows;
split is checked again here). Labels are read only for train/val ids (split filtered before reading label rows).
Writes data/trval_e12.npy (float16 [N,12,5,448]), data/trval_bones.npy (float16 [N,17,5,448]), data/trval_meta.npz."""
import numpy as np, pandas as pd, os, zipfile
R = "/home/advantech/Desktop/Tony/VLM/OWL/"
O = R + "mw_exp/outputs_final/organ_scope/data/"; os.makedirs(O, exist_ok=True)
spl = pd.read_csv(R + "data/split.csv").set_index("study_id")["split"]
ok = set(spl[spl.isin(["train", "val"])].index)
f = R + "mw_exp/outputs_final/results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
z = np.load(f, mmap_mode="r")
ids = list(z["ids"]); units = list(z["units"]); print(len(ids), units)
keep = np.array([i for i, s in enumerate(ids) if s in ok]); assert len(keep) == len(ids), "test rows present?"
ids = [ids[i] for i in keep]
E = z["E"]; print(E.dtype, E.shape)
np.save(O + "trval_e12.npy", np.ascontiguousarray(E[keep][:, :12]).astype(np.float16))
np.save(O + "trval_bones.npy", np.ascontiguousarray(E[keep][:, 12:]).astype(np.float16)); del E
P = z["P"][keep]
lid = pd.read_csv(R + "data/labels_disease14.csv", usecols=["study_id"])["study_id"].tolist()
skip = [i + 1 for i, s in enumerate(lid) if s not in ok]
lab = pd.read_csv(R + "data/labels_disease14.csv", skiprows=skip).set_index("study_id")
assert set(lab.index) <= ok
lab = lab.reindex(ids)
meta = pd.read_csv(R + "../merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")
age = meta.reindex(ids)["Age"].values.astype(float)
np.savez(O + "trval_meta.npz", ids=np.array(ids), split=spl.reindex(ids).values.astype(str), P=P, units=np.array(units),
         findings=np.array(lab.columns), Y=lab.values.astype(np.float32), age=age)
print("N", len(ids), pd.Series(spl.reindex(ids).values).value_counts().to_dict(), "pos", dict(zip(lab.columns, np.nansum(lab.values, 0))))
