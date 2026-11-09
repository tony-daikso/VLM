"""spine_ft / pelvis_ft labels + crop priority (train/val only; test ids dropped before any label line is parsed).
Writes outputs_fracture/spine_ft/in/labels.csv: study_id, split, y (overall fracture), t_vert, t_pel, t_rib (site labels:
1 = fracture at that site, 0 = fracture-free, NaN = fracture elsewhere / unclear), fold (= sitewise outer StratifiedKFold
(5, shuffle, rs=0) on the sitewise common train frame; train rows outside it get fold = hash % 5; val = -1), tierA
(train: all fracture positives + 4000 random negatives, rs=0), prio (crop order: tierA train -> val -> rest of train)."""
import sys, zlib, numpy as np, pandas as pd
from pathlib import Path
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from sklearn.model_selection import StratifiedKFold
df, _ = S.build_frame()
assert set(df.split) <= {"train", "val"}
need = sorted({c for fs in S.FEATS.values() for v in fs.values() for c in v} | set(S.C3M))
tr = df[df.split == "train"].dropna(subset=need)
fold = pd.Series(-1, index=df.index)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, tr["y"])):
    fold[tr.index[b]] = k
rest = (df.split == "train") & (fold == -1)
fold[rest] = [zlib.crc32(s.encode()) % 5 for s in df.index[rest]]
out = pd.DataFrame({"study_id": df.index, "split": df.split.values, "y": df.y.values, "t_vert": df.t_vert.values,
                    "t_pel": df.t_pel.values, "t_rib": df.t_rib.values, "age": df.age.values, "fold": fold.values,
                    "in_sitewise": df.index.isin(tr.index) | (df.split == "val").values})
rng = np.random.default_rng(0)
trn = out[(out.split == "train") & (out.y == 0)].study_id.values
negA = set(rng.choice(np.sort(trn), 4000, replace=False))
out["tierA"] = (out.split == "train") & ((out.y == 1) | out.study_id.isin(negA))
out["prio"] = np.where(out.tierA, 0, np.where(out.split == "val", 1, 2))
out = out.sample(frac=1, random_state=0).sort_values("prio", kind="stable")
out.to_csv(MW / "outputs_fracture/spine_ft/in/labels.csv", index=False)
out.study_id.to_csv(MW / "outputs_fracture/spine_ft/in/ids_prio.txt", index=False, header=False)
print(out.groupby(["prio", "split"]).agg(n=("y", "size"), pos=("y", "sum"), vert=("t_vert", "sum"), pel=("t_pel", "sum")))
print("fold counts", out[out.split == "train"].groupby("fold").size().to_dict())
