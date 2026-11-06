"""GS6 step 1 (CPU): for GB-missing/tiny cases (3 mm GB < 100 vox) compute a world-space (RAS mm) crop box for the raw CT:
liver bbox (3 mm prep units) U expected-GB-fossa point, expanded by 45 mm. No liver (<1000 vox) -> full volume (box empty).
usage: gs6_boxes.py ids.txt prep_dir[,dir2..] out.csv"""
import sys, os, numpy as np, pandas as pd
EXP = np.array([0.3355, 0.7002, 0.6623]); MARGIN = 45.0
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]; PD = sys.argv[2].split(","); rows = []
for s in ids:
    fp = next((f"{d}/{s}.npz" for d in PD if os.path.exists(f"{d}/{s}.npz")), None)
    if fp is None: rows.append(dict(study_id=s, status="noprep")); continue
    z = np.load(fp); names = list(z["unit_names"]); u = z["units"]; A = z["affine"]
    L = u == names.index("liver") + 1; gbv = int((u == names.index("gallbladder") + 1).sum())
    r = dict(study_id=s, gb_vox3=gbv, liver_vox3=int(L.sum()), status="full")
    if L.sum() >= 1000:
        kz, jy, ix = np.where(L)
        lo = np.array([kz.min(), jy.min(), ix.min()], float); hi = np.array([kz.max(), jy.max(), ix.max()], float)
        pt = lo + EXP * (hi - lo)
        pts_zyx = np.array([lo, hi, pt])
        ijk = pts_zyx[:, ::-1]                                   # (z,y,x) array -> nifti (i=x, j=y, k=z)
        w = (A @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
        wmin, wmax = w.min(0) - MARGIN, w.max(0) + MARGIN
        r.update(status="box", **{f"wmin_{a}": float(v) for a, v in zip("xyz", wmin)}, **{f"wmax_{a}": float(v) for a, v in zip("xyz", wmax)})
    rows.append(r)
pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
print("BOXES_DONE", pd.DataFrame(rows).status.value_counts().to_dict(), flush=True)
