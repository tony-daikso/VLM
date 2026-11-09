"""rib_ft2 post (Spark CPU): assemble OOF (5 folds) + val (full-train model), then choose by TRAIN OOF (t_rib) among
probe (rib_ft final), ft2, and mean of train-z scores -> ~/OWL/rib_ft2/out/{oof_train.csv,val_scores.csv,summary.json} (col ft).
Val labels never read."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
B = "/home/daikso_spark/OWL"; lab = pd.read_csv(f"{B}/rib_ft/labels.csv").set_index("study_id"); assert set(lab.split) <= {"train", "val"}
o2 = pd.concat([pd.read_csv(f"{B}/rib_ft2/runs/main/oof_fold{k}.csv") for k in range(5)]).set_index("study_id")["ft"]
v2 = pd.read_csv(f"{B}/rib_ft2/runs/main/val_scores_seed0.csv").set_index("study_id")["ft"]
o1 = pd.read_csv(f"{B}/rib_ft/final/oof_train.csv").set_index("study_id")["ft"]; v1 = pd.read_csv(f"{B}/rib_ft/final/val_scores.csv").set_index("study_id")["ft"]
z = lambda s, ref: (s - ref.mean()) / ref.std()
cand = {"probe": (o1, v1), "ft2": (z(o2, o2), z(v2, o2)), "mean": ((z(o1, o1) + z(o2, o2).reindex(o1.index)) / 2, (z(v1, o1) + z(v2, o2).reindex(v1.index)) / 2)}
res = {}
for k, (o, v) in cand.items():
    y = lab.loc[o.index, "t_rib"]; m = y.notna() & o.notna(); res[k] = float(auc(y[m], o[m]))
best = max(res, key=res.get); o, v = cand[best]
import os; os.makedirs(f"{B}/rib_ft2/out", exist_ok=True)
pd.DataFrame({"study_id": o.index, "ft": o.values}).dropna().to_csv(f"{B}/rib_ft2/out/oof_train.csv", index=False)
json.dump({"train_oof_t_rib": res, "selected": best, "n_train": int(o.notna().sum()), "n_val": int(v.notna().sum())}, open(f"{B}/rib_ft2/out/summary.json", "w"), indent=1)
pd.DataFrame({"study_id": v.index, "ft": v.values}).dropna().to_csv(f"{B}/rib_ft2/out/val_scores.csv", index=False)
print(json.dumps(res), "selected", best)
