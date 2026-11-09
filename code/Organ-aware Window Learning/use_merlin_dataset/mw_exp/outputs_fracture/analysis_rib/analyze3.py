import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); from analyze_util import auc
OUT = Path(__file__).parent
d = pd.read_pickle(OUT / "table_parsed.pkl"); tr = d[d.split == "train"]
neg = tr[tr.y == 0]; pos = tr[tr.rib == 1].copy()
thr = np.quantile(neg.s_bone_ens, .9)
r = {}
r["sens90_rib_only"] = float((pos[~pos.other_site_rib].s_bone_ens >= thr).mean()); r["sens90_rib_with_other"] = float((pos[pos.other_site_rib].s_bone_ens >= thr).mean())
def minvox(x):
    if not x.rib_levels or np.isnan(x.nz): return np.nan
    v = []
    for t in x.rib_levels.split(";"):
        sd, lv = t[0], int(t[1:])
        v += [x[f"v_rib_{s}_{lv}"] for s in (["left", "right"] if sd in "B?" else [{"L": "left", "R": "right"}[sd]])]
    return np.nanmax(v)
pos["rep_vox"] = pos.apply(minvox, axis=1)
pos["vox_bin"] = pd.cut(pos.rep_vox, [-1, 99, 300, 700, 1e9], labels=["<100（缺）", "100–300", "300–700", ">700"])
rows = []
for g, gd in pos.groupby("vox_bin", observed=True):
    rows.append(dict(子群=str(g), n=len(gd), AUROC=round(auc(np.r_[np.ones(len(gd)), np.zeros(len(neg))], np.r_[gd.s_bone_ens, neg.s_bone_ens]), 3)))
r["reported_rib_voxels"] = rows
r["rep_vox_median"] = float(pos.rep_vox.median())
# rib voxel totals: neg vs pos ; rib-mask volume as a feature
r["auc_ribvox_total_rib"] = auc(tr.rib, tr.rib_vox_total)
r["auc_n_rib_units_rib"] = auc(tr.rib, tr.n_rib_units_present)
r["auc_zmm_rib"] = auc(tr.rib, tr.z_mm)
low = pos.sort_values("s_bone_ens").head(15)[["cat", "p", "acute", "sites", "rib_levels", "fov", "rib_age", "rib_sent"]]
print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
pd.set_option("display.max_colwidth", 160); pd.set_option("display.width", 250)
print(low.to_string(index=False))
json.dump(r, open(OUT / "results_rib3.json", "w"), ensure_ascii=False, indent=1, default=float)
