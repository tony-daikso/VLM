"""combine rib_probe3 / pv15_probe runs -> OUT/oof_train.csv, val_scores.csv (column ft), summary.json.
train = own-fold OOF; val = mean of the 5 fold models; runs = mean of per-run train-z-scored scores. Only train labels read.
usage: combine3.py LABELS TARGET COL OUT RUN [RUN ...]   (COL e.g. ft_side; TARGET e.g. t_rib)"""
import sys, json, os, pandas as pd
from sklearn.metrics import roc_auc_score as auc
lab = pd.read_csv(sys.argv[1]).set_index("study_id"); assert set(lab.split) <= {"train", "val"}
tgt, col, out, runs = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]; os.makedirs(out, exist_ok=True); TR, VA, info = [], [], {}
for r in runs:
    s = pd.concat([pd.read_csv(f"{r}/score_fold{k}.csv") for k in range(5)])
    t = s[s.split == "train"].set_index("study_id")[col]; assert t.index.is_unique
    v = s[s.split == "val"].groupby("study_id")[col].agg(["mean", "count"])
    mu, sd = t.mean(), t.std(); TR.append((t - mu) / sd); VA.append((v["mean"] - mu) / sd)
    y = lab.loc[t.index, tgt]; m = y.notna(); info[r] = float(auc(y[m], t[m]))
tr = pd.concat(TR, axis=1).mean(1); va = pd.concat(VA, axis=1).mean(1)
y = lab.loc[tr.index, tgt]; m = y.notna()
res = {"runs": runs, "col": col, "per_run_oof": info, "oof_all": float(auc(y[m], tr[m])), "oof_overall_fx": float(auc(lab.loc[tr.index, "y"], tr)),
       "n_train": int(len(tr)), "n_val": int(len(va))}
pd.DataFrame({"study_id": tr.index, "ft": tr.values}).to_csv(f"{out}/oof_train.csv", index=False)
pd.DataFrame({"study_id": va.index, "ft": va.values}).to_csv(f"{out}/val_scores.csv", index=False)
json.dump(res, open(f"{out}/summary.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
