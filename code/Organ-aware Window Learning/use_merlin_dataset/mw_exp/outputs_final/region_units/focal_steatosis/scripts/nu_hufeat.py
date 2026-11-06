"""Exp N: per-crop HU scalars of the ath unit (nu_crop.py hu_feats [K,R,5] = n>=130, n>=300, max, p99, median) ->
csv with the vessel-wall-shell (r0) n>=300 HU voxels, max and p99 of each crop (n>=130 is dominated by contrast lumen,
not used). Absent crops -> NaN. usage: nu_hufeat.py crops_dir out.csv"""
import sys, glob, os, numpy as np, pandas as pd
NM = ["infrarenal", "bifurcation", "cia_right", "cia_left"]; rows = []
for f in glob.glob(sys.argv[1] + "/*.npz"):
    if ".tmp" in f: continue
    z = np.load(f); d = {"study_id": os.path.basename(f)[:-4]}
    hf = z["hu_feats"] if "hu_feats" in z else np.full((4, 2, 5), np.nan); pr = z["present"]
    for k, n in enumerate(NM):
        v = hf[k, 0] if pr[k] else np.full(5, np.nan)
        d[f"nh_{n}_n300"], d[f"nh_{n}_max"], d[f"nh_{n}_p99"] = v[1], v[2], v[3]
    rows.append(d)
pd.DataFrame(rows).to_csv(sys.argv[2], index=False); print("HUFEAT_DONE", len(rows))
