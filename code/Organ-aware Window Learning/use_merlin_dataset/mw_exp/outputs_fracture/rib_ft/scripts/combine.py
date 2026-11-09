"""combine rib_ft runs (score_fold*.csv from rib_ft.py --score) -> OUT/oof_train.csv, val_scores.csv, summary.json.
train row score = its own fold's model (OOF); val = mean over the 5 fold models; several runs (seeds) = mean of
per-run train-z-scored scores. Only train labels are read for the OOF AUROC (val labels never touched here).
usage: combine.py LABELS OUT RUN [RUN ...]"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
lab = pd.read_csv(sys.argv[1]).set_index("study_id"); assert set(lab.split) <= {"train", "val"}
out = sys.argv[2]; runs = sys.argv[3:]; TR, VA, info = [], [], {}
for r in runs:
    s = pd.concat([pd.read_csv(f"{r}/score_fold{k}.csv") for k in range(5)])
    t = s[s.split == "train"].set_index("study_id")["ft"]; assert t.index.is_unique
    assert (lab.loc[t.index, "fold"].values == s[s.split == "train"]["fold"].values).all()
    v = s[s.split == "val"].groupby("study_id")["ft"].agg(["mean", "count"]); assert (v["count"] == 5).all()
    mu, sd = t.mean(), t.std(); TR.append((t - mu) / sd); VA.append((v["mean"] - mu) / sd)
    y = lab.loc[t.index, "t_rib"]; m = y.notna(); info[r] = float(auc(y[m], t[m]))
tr = pd.concat(TR, axis=1).mean(1); va = pd.concat(VA, axis=1).mean(1)
y = lab.loc[tr.index, "t_rib"]; m = y.notna(); t0 = (lab.loc[tr.index, "tier"] == 0) & m
res = {"runs": runs, "per_run_oof_t_rib": info, "oof_t_rib_all": float(auc(y[m], tr[m])), "oof_t_rib_tier0": float(auc(y[t0], tr[t0])),
       "oof_overall_fx": float(auc(lab.loc[tr.index, "y"], tr)), "n_train": int(len(tr)), "n_val": int(len(va))}
pd.DataFrame({"study_id": tr.index, "ft": tr.values}).to_csv(f"{out}/oof_train.csv", index=False)
pd.DataFrame({"study_id": va.index, "ft": va.values}).to_csv(f"{out}/val_scores.csv", index=False)
json.dump(res, open(f"{out}/summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
