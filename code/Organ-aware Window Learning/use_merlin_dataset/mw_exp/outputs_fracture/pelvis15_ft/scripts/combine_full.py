"""pelvis15_ft: average the seed runs (ft logits) -> final/{oof_train.csv,val_scores.csv,summary.json}. Train OOF metrics only
(t_pel on pool rows and on all train rows; overall fracture label); val scores written, never scored here. Paired bootstrap
(2000) on all train rows (t_pel known) vs PV15 (pelvis15/final_m400) and U_org_pel (ref_train.csv).
usage: combine_full.py OUT RUN_DIR [RUN_DIR ...]"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score as auc
P = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); runs = [Path(r) for r in sys.argv[2:]]
lab = pd.read_csv(P / "labels.csv"); assert set(lab.split) <= {"train", "val"}
tl = lab[lab.split == "train"].set_index("study_id")
pool = {l.strip() for l in open(out.parent / "pool_ids.txt") if l.strip()}
O = [pd.read_csv(r / "oof_train.csv").set_index("study_id").ft.rename(r.name) for r in runs]
V = [pd.read_csv(r / "val_scores.csv").set_index("study_id").ft.rename(r.name) for r in runs]
o = pd.concat(O, axis=1); v = pd.concat(V, axis=1)
res = {"runs": [str(r) for r in runs], "per_run": {}}
d = tl.join(o)
for c in o.columns:
    m = d.t_pel.notna() & d[c].notna(); mp = m & d.index.isin(pool)
    res["per_run"][c] = {"oof_pool": auc(d.t_pel[mp], d[c][mp]), "oof_all": auc(d.t_pel[m], d[c][m]), "n_all": int(m.sum())}
of = o.mean(1, skipna=False); vf = v.mean(1, skipna=False)
pd.DataFrame({"study_id": of.index, "ft": of.values}).dropna().to_csv(out / "oof_train.csv", index=False)
pd.DataFrame({"study_id": vf.index, "ft": vf.values}).dropna().to_csv(out / "val_scores.csv", index=False)
d = tl.join(of.rename("FT")).join(pd.read_csv(P / "final_m400/oof_train.csv").set_index("study_id").ft.rename("PV15")) \
      .join(pd.read_csv(P / "ref_train.csv", index_col=0)[["U_org_pel", "U_pel"]])
m = d.t_pel.notna() & d.FT.notna(); mp = m & d.index.isin(pool); my = d.y.notna() & d.FT.notna()
res.update(oof_pool=auc(d.t_pel[mp], d.FT[mp]), n_pool=int(mp.sum()), oof_all=auc(d.t_pel[m], d.FT[m]), n_all=int(m.sum()),
           n_pos=int(d.t_pel[m].sum()), oof_overall_fx=auc(d.y[my], d.FT[my]), n_train_scored=int(of.notna().sum()), n_val=int(vf.notna().sum()))
q = d[m & d.PV15.notna() & d.U_org_pel.notna()]; y = q.t_pel.values.astype(int); rng = np.random.default_rng(0)
B = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[b].sum() < len(b)]
for ref in ("PV15", "U_org_pel", "U_pel"):
    dv = np.array([auc(y[b], q.FT.values[b]) - auc(y[b], q[ref].values[b]) for b in B])
    res[f"vs_{ref}"] = {"ref_auc": auc(y, q[ref]), "ft_auc": auc(y, q.FT), "delta": auc(y, q.FT) - auc(y, q[ref]),
                        "ci": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))], "n": int(len(y))}
json.dump(res, open(out / "summary.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
