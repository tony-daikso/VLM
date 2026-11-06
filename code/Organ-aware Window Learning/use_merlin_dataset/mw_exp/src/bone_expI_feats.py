"""Exp I: derived stack members in the OOF member format (ids, split, logit, y) from OOF member files (train = OOF,
val = mean of fold models; nothing fit here).
  site  : --src glob --site k        -> seed-mean site logit k (0 vertebra / 1 rib / 2 pelvis), e.g. Exp G pelvis site
  peak  : --src glob                 -> per-level vertebra peak = max - median over present T8..L5 instance logits (seed mean)
usage: python bone_expI_feats.py site --src '.../oof_bone_E_ref__site_max-rib18__mean__all3*.npz' --site 2 --out X.npz
       python bone_expI_feats.py peak --src '.../expI/members/oof_bone_E_lv1_s*.npz' --out Y.npz
"""
import argparse, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["site", "peak"]); ap.add_argument("--src", required=True)
ap.add_argument("--site", type=int, default=2); ap.add_argument("--out", required=True)
a = ap.parse_args()
fs = sorted(glob.glob(a.src)); assert fs
V, ids, split, y = [], None, None, None
for f in fs:
    z = np.load(f, allow_pickle=True)
    assert set(np.unique(z["split"])) <= {"train", "val"}
    if ids is None:
        ids, split, y = z["ids"], z["split"], z["y"]
    assert (z["ids"] == ids).all()
    if a.kind == "site":
        V.append(z["site_logit"][:, a.site])
    else:
        I = z["inst"][:, :10]                       # first 10 instance columns = T8..L5 (NaN = absent)
        assert [str(u) for u in z["inst_units"][:10]] == ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
        V.append(I)
v = np.nanmean(np.stack(V), 0) if a.kind == "peak" else np.mean(V, 0)
if a.kind == "peak":
    with np.errstate(all="ignore"):
        v = np.nanmax(v, 1) - np.nanmedian(v, 1)
    v = np.where(np.isnan(v), 0.0, v)
np.savez(a.out, ids=ids, split=split, logit=v.astype(np.float32), y=y, exp=f"{a.kind}:{a.src}", n_files=len(fs))
print(a.out, len(fs), "files; train/val non-nan:", int(np.isfinite(v[split == "train"]).sum()), int(np.isfinite(v[split == "val"]).sum()))
