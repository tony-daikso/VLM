"""Combine rib_straight full runs (runs/<run>/fold{k}.npz; each seed a run dir) -> OUT/{oof_train.csv, val_scores.csv, summary.json}.
Per seed: train OOF = held-out fold scores; val = mean of the 5 fold models. Seeds averaged after per-seed z-scoring on train.
Train-only metrics (t_rib OOF, all train); val written to file but NOT scored here.
usage: rs_combine.py OUT RUN [RUN ...]"""
import sys, os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
RS = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/rib_straight")
out = sys.argv[1]; runs = sys.argv[2:]; os.makedirs(out, exist_ok=True)
lab = pd.read_csv(RS + "/in/labels.csv").set_index("study_id"); assert set(lab.split) <= {"train", "val"}
TR, VA, per = [], [], {}
for r in runs:
    d = RS + "/runs/" + r; b, v = {}, {}
    fs = [f"{d}/fold{k}.npz" for k in range(5)]
    if not all(os.path.exists(f) for f in fs): print(r, "incomplete, skipped", [os.path.exists(f) for f in fs]); continue
    for f in fs:
        z = np.load(f)
        b.update(zip(z["b_ids"].astype(str), z["pb"]))
        if "v_ids" in z.files:
            for s, p in zip(z["v_ids"].astype(str), z["pv"]): v.setdefault(s, []).append(p)
    tb = pd.Series(b); vb = pd.Series({s: np.mean(p) for s, p in v.items() if len(p) == 5})
    mu, sd = tb.mean(), tb.std(); TR.append(((tb - mu) / sd).rename(r)); VA.append(((vb - mu) / sd).rename(r))
    t = lab.reindex(tb.index); m = t.t_rib.notna() & (t.split == "train")
    per[r] = float(auc(t.t_rib[m], tb[m]))
tr = pd.concat(TR, axis=1).mean(1); va = pd.concat(VA, axis=1).mean(1)
t = lab.reindex(tr.index); m = t.t_rib.notna() & (t.split == "train")
res = dict(runs=list(per), per_run_oof_t_rib=per, oof_t_rib=float(auc(t.t_rib[m], tr[m])), n_train=int(len(tr)), n_val=int(len(va)),
           oof_overall=float(auc(t.y[t.y.notna()], tr[t.y.notna()])))
pd.DataFrame({"study_id": tr.index, "ft": tr.values}).to_csv(out + "/oof_train.csv", index=False)
pd.DataFrame({"study_id": va.index, "ft": va.values}).to_csv(out + "/val_scores.csv", index=False)
json.dump(res, open(out + "/summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
