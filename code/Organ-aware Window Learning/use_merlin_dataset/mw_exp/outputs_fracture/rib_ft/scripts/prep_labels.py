"""rib_ft labels: train/val only (split.csv filtered BEFORE labels are parsed, via sitewise.build_frame); test never read.
Columns: study_id, split, y (overall fracture), t_rib (rib fx=1 / fracture-free=0 / other-site fx=NaN), rib_any (rib in sites),
age, fold (train: sitewise outer StratifiedKFold(5, rs=0) on its common train set; other train rows: md5 % 5; val: -1), tier (ids_prio)."""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts"))
import sitewise as S
df, _ = S.build_frame()
assert set(df.split) <= {"train", "val"}
need = sorted({c for fs in S.FEATS.values() for v in fs.values() for c in v} | set(S.C3M))
tr = df[df.split == "train"].dropna(subset=need).copy()
fold = pd.Series(-1, index=df.index)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, tr["y"])):
    fold[tr.index[b]] = k
m = (df.split == "train") & (fold < 0)
fold[m] = [int(hashlib.md5(s.encode()).hexdigest(), 16) % 5 for s in df.index[m]]
pr = pd.read_csv(MW / "outputs_ribs15/ids_prio.csv").set_index("study_id")["tier"]
out = pd.DataFrame({"study_id": df.index, "split": df.split.values, "y": df.y.values, "t_rib": df.t_rib.values,
                    "rib_any": df.sites.apply(lambda s: "rib" in s).values, "age": df.age.values, "fold": fold.values,
                    "in_sitewise": df.index.isin(tr.index) | (df.split == "val").values, "tier": df.index.map(pr).values})
out.to_csv(MW / "outputs_fracture/rib_ft/labels.csv", index=False)
print(out.groupby(["split", "tier"]).agg(n=("y", "size"), y=("y", "sum"), rib=("t_rib", "sum")))
print(out[out.split == "train"].groupby("fold").agg(n=("y", "size"), y=("y", "sum"), rib=("t_rib", "sum")))
