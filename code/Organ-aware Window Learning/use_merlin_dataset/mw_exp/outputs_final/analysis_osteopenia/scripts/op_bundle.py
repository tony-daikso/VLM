"""Collect train/val-only inputs (bone geometry/HU from fracture_features, aorta HU from gallstones analysis, metadata) into one CSV."""
import glob, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/"; A = R + "outputs_final/analysis_osteopenia/"
TRV = set(pd.read_csv(A + "labels_trval.csv").study_id)
fr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "outputs_fracture/fracture_features_*.csv"))])
fr = fr[fr.id.isin(TRV)].drop_duplicates("id").rename(columns={"id": "study_id"})
ao = pd.concat([pd.read_csv(R + "outputs_final/analysis_gallstones/" + f, usecols=["study_id", "aorta_hu_med", "liver_hu_med", "spleen_hu_med"]) for f in ["hu_a6.csv", "hu_ct.csv"]])
ao = ao[ao.study_id.isin(TRV)].drop_duplicates("study_id")
m = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").rename(columns={"study id": "study_id"})
m = m[m.study_id.isin(TRV)].drop_duplicates("study_id")[["study_id", "Age", "Gender", "contrast", "phase", "manufacturer", "kvp", "slicethickness"]]
b = m.merge(fr, on="study_id", how="left").merge(ao, on="study_id", how="left")
b.to_csv(A + "bundle_trval.csv", index=False); print(b.shape, b.L1_hu.notna().mean(), b.aorta_hu_med.notna().mean())
