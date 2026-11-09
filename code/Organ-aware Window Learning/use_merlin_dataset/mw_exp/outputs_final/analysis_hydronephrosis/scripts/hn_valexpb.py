"""Extract expB E0/E3_late val probabilities for hydronephrosis + trval metadata (val/train ids only; test never read out)."""
import numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"; O = R + "OWL/mw_exp/outputs_final/"; A = O + "analysis_hydronephrosis/"
lab = pd.read_csv(A + "labels_trval.csv"); ok = set(lab.study_id)
v = pd.DataFrame({"study_id": sorted(lab[lab.split == "val"].study_id)})
for nm in ["E0", "E3_late"]:
    z = np.load(O + f"results/expB/{nm}.npz", allow_pickle=True); fi = list(z["findings"]).index("hydronephrosis")
    m = np.array([s == "val" for s in z["split"]])
    pv = pd.Series(z["prob"].mean(0)[m, fi], index=np.array(z["ids"])[m]); v["expB_" + nm] = v.study_id.map(pv)
v.to_csv(A + "expB_val_probs.csv", index=False)
meta = pd.read_csv(R + "merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta[meta.study_id.isin(ok)][["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "slicethickness"]].to_csv(A + "meta_trval.csv", index=False)
print(v.notna().sum().to_dict())
