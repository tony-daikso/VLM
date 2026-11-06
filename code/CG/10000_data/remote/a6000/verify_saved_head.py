"""Reload a head saved by probe.py --save-model, re-run it on the VAL split only (from the raw cached embedding
matrix, standardized with the saved stats) and compare with (a) the training-time predictions and (b) the
reference run (expB tables_val.md). Test rows are never scored."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from infer_head import load_model, predict
from probe import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--matrix", required=True)
ap.add_argument("--train-npz", required=True, help="results/<subdir>/E3_late.npz of the save-model run")
ap.add_argument("--ref-npz", required=True, help="results/expB/E3_late.npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ck, models = load_model(a.model)
meta = ck["meta"]
spl = pd.read_csv(meta["split_file"]).set_index("study_id")["split"]
lab = pd.read_csv(meta["labels_file"]).set_index("study_id")
z = np.load(a.matrix, mmap_mode="r")
ids = np.array(z["ids"]); units = list(z["units"])
assert units == meta["units"]
va = np.where(spl.reindex(ids).values == "val")[0]
E = torch.from_numpy(np.asarray(z["E"][va]).astype(np.float32)); P = torch.from_numpy(np.asarray(z["P"][va]).astype(np.float32))
pr = predict(models, ck, E, P)                                      # [S,Nv,F]
vid = ids[va]
Y = lab.loc[vid, meta["findings"]].values.astype(np.float32)
tr = np.load(a.train_npz); ref = np.load(a.ref_npz)
def val_prob(r):
    m = {c: i for i, c in enumerate(r["ids"])}; ix = [m[c] for c in vid]
    assert list(r["findings"]) == meta["findings"]
    return r["prob"][:, ix]
ptr, pref = val_prob(tr), val_prob(ref)
rows = []
for f, name in enumerate(meta["findings"]):
    rows.append({"finding": name, "val_pos": int(np.nansum(Y[:, f])),
                 "reload": auroc(Y[:, f], pr.mean(0)[:, f]), "train_time": auroc(Y[:, f], ptr.mean(0)[:, f]),
                 "expB_ref": auroc(Y[:, f], pref.mean(0)[:, f]),
                 "reload_seed_mean": float(np.mean([auroc(Y[:, f], p[:, f]) for p in pr]))})
df = pd.DataFrame(rows)
df.loc[len(df)] = {"finding": "MEAN", "val_pos": -1, **{c: df[c].mean() for c in ["reload", "train_time", "expB_ref", "reload_seed_mean"]}}
df.to_csv(a.out, index=False, float_format="%.4f")
print(df.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"N_val={len(va)}  max|reload - train_time prob| = {np.abs(pr - ptr).max():.2e}")
