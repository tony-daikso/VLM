"""Export train/val-only base tables for the integrated evaluation (run on A6000).
- split filter FIRST, then labels (labels_disease14.csv rows outside train/val are dropped on read).
- expB E3_late: only the val rows are kept (the test rows are sliced off immediately and never used).
Writes work/base/{labels14_trval.csv, age_trval.csv, expB_E3late_val.csv}."""
import os, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/"
O = R + "mw_exp/outputs_final/integrated_best/work/base/"; os.makedirs(O, exist_ok=True)
spl = pd.read_csv(R + "data/split.csv"); spl = spl[spl.split.isin(["train", "val"])]
keep = set(spl.study_id)
lab = pd.concat([c[c.study_id.isin(keep)] for c in pd.read_csv(R + "data/labels_disease14.csv", chunksize=2000)])
lab = spl.merge(lab, on="study_id", how="inner"); lab.to_csv(O + "labels14_trval.csv", index=False)
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv")
meta = meta.rename(columns={"study id": "study_id"}).drop_duplicates("study_id")
meta = meta[meta.study_id.isin(keep)][["study_id", "Age"]]; meta.to_csv(O + "age_trval.csv", index=False)
z = np.load(R + "mw_exp/outputs_final/results/expB/E3_late.npz", allow_pickle=True)
sp = z["split"]; vm = sp == "val"
prob = z["prob"][:, vm, :].mean(0)          # seed ensemble, val rows only
ids = z["ids"][vm]; fnd = list(z["findings"])
del z
pd.DataFrame(prob, columns=fnd).assign(study_id=ids)[["study_id"] + fnd].to_csv(O + "expB_E3late_val.csv", index=False)
print(len(lab), lab.split.value_counts().to_dict(), "age", len(meta), "val", vm.sum())
