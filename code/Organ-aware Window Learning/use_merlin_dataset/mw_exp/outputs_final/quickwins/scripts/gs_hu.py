"""GB HU features from prep npz (3mm HU + unit label map). Usage: gs_hu.py ids.txt prep_dir out.csv nproc"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage
ids, prep, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz")
    except Exception:
        return None
    names = list(z["unit_names"]); hu = z["hu"].astype(np.float32); u = z["units"]
    d = {"study_id": sid}
    gi = names.index("gallbladder") + 1
    m = u == gi
    d["gb_vox"] = int(m.sum())
    for nm, k in (("liver", "liver"), ("aorta", "aorta"), ("spleen", "spleen")):
        mm = u == names.index(k) + 1
        d[f"{nm}_hu_med"] = float(np.median(hu[mm])) if mm.sum() > 50 else np.nan
    if d["gb_vox"] >= 5:
        v = hu[m]
        q = np.percentile(v, [5, 10, 25, 50, 75, 90, 95, 99])
        for p, x in zip([5, 10, 25, 50, 75, 90, 95, 99], q):
            d[f"gb_p{p}"] = float(x)
        d["gb_max"] = float(v.max()); d["gb_mean"] = float(v.mean()); d["gb_std"] = float(v.std())
        for t in (60, 80, 100, 130, 200):
            d[f"gb_n{t}"] = int((v > t).sum())
        # hyperdense voxels relative to GB median (contrast-robust)
        d["gb_n_med40"] = int((v > np.median(v) + 40).sum())
        d["gb_n_lowfat"] = int((v < -20).sum())  # fat/gas (cholesterol / gas-containing stones)
        # 1-voxel dilated shell: stones abutting the wall may be excluded from the TS mask
        dl = ndimage.binary_dilation(m, iterations=1) & ~m & ((u == 0) | (u == gi))
        sv = hu[dl]
        d["shell_n100"] = int((sv > 100).sum()); d["shell_p99"] = float(np.percentile(sv, 99)) if sv.size else np.nan
        # connected hyperdense blob size inside GB
        lab, n = ndimage.label(m & (hu > 100))
        d["gb_maxblob100"] = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    return d
if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L))
