"""Extra rows for the two user-approved label splits (脂肪肝 -> 瀰漫性/局部; 闌尾炎 -> 確定急性/可疑等).
The split agents' outputs are reused as-is (their train-OOF model choice); here the variant with the highest train OOF
is taken and metrics are recomputed with ib_core.evaluate. Baseline = expB E3_late of the ORIGINAL finding (official
model) scored against the new label; the split agents' re-trained E3_late on the new label is reported as 'rerun'.
usage: ib_split.py <ib_root> <key> <orig_finding> <oof.csv> <val.csv> <zh name> [primary(0/1)]"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_core import evaluate, auc, sens_at_spec
R, key, fd, fo, fv, zh = sys.argv[1].rstrip("/") + "/", *sys.argv[2:7]
primary = len(sys.argv) > 7 and sys.argv[7] == "1"
age = pd.read_csv(R + "base/age_trval.csv"); exb = pd.read_csv(R + "base/expB_E3late_val.csv")[["study_id", fd]].rename(columns={fd: "expB_E3"})
o = pd.read_csv(fo); v = pd.read_csv(fv)
o = o[o.y.notna()]; v = v[v.y.notna()]
cands = [c for c in o.columns if c not in ("study_id", "fold", "y") and c in v.columns]
var = {c: auc(o.y, o[c]) for c in cands}; ch = max(var, key=var.get)
T = o.merge(age, on="study_id", how="left"); V = v.merge(exb, on="study_id").merge(age, on="study_id", how="left")
_, thr = sens_at_spec(T.y.values.astype(int), T[ch].values)
r = evaluate(key, T.y.values, T[ch].values, T.E3_late.values, T.Age.values, V.y.values, V[ch].values, V.expB_E3.values,
             V.E3_late.values, V.Age.values, thr_oof=thr)
r.update(featset=ch, cand="split agent (train-OOF choice)", cv=var, zh=zh, orig=fd)
os.makedirs(R + "split", exist_ok=True); os.makedirs(R + "results", exist_ok=True)
json.dump(r, open(R + f"results/split_{key}.json", "w"), indent=1, default=float)
pd.DataFrame({"study_id": V.study_id, "y": V.y.astype(int), "expB_E3_late": V.expB_E3, "stack": V[ch]}).to_csv(R + f"split/{key}_val.csv", index=False)
json.dump(dict(key=key, zh=zh, substitutes=fd if primary else None, primary=primary), open(R + f"split/{key}.json", "w"))
print(key, ch, f"OOF {r['oof']:.4f} val {r['val']:.4f} {np.round(r['val_ci'],3)} expB {r['val_E3_expB']:.4f} rerun {r['val_E3_rerun']:.4f} d {r['d_val_expB']['mean']:+.4f} {np.round(r['d_val_expB']['ci'],3)} age {r['val_age']:.3f}/{r['val_age_E3']:.3f}")
