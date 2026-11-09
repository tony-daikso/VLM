"""Aorto-iliac wall calcification features from prep npz (3 mm HU) + TS fast multilabel mask (same 3 mm grid).
Usage: ath_hu.py ids.txt prep_dir mask_dir out.csv nproc"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage as ndi
ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
AO, ILL, ILR = 52, 65, 66
B1 = ndi.generate_binary_structure(3, 1)
def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz"); hu = z["hu"]
        m = nib.load(f"{mdir}/{sid}.nii.gz")
        lab = np.asanyarray(m.dataobj).astype(np.uint8).transpose(2, 1, 0)
    except Exception:
        return None
    if lab.shape != hu.shape:
        return {"study_id": sid, "grid_mismatch": 1}
    d = {"study_id": sid, "grid_mismatch": 0}
    V = np.isin(lab, [AO, ILL, ILR])
    d["a_vox"] = int((lab == AO).sum()); d["il_vox"] = int(np.isin(lab, [ILL, ILR]).sum())
    if V.sum() < 10:
        return d
    idx = np.argwhere(V); lo = np.maximum(idx.min(0) - 4, 0); hi = idx.max(0) + 5
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    L = lab[sl]; H = hu[sl].astype(np.float32)
    A = L == AO; IL = np.isin(L, [ILL, ILR]); V = A | IL
    free = (L == 0) | V
    core = ndi.binary_erosion(A, B1, iterations=1)
    if core.sum() < 20: core = A
    lv = H[core] if core.sum() else np.array([np.nan])
    d["lum_p10"], d["lum_p50"], d["lum_p90"] = [float(x) for x in np.nanpercentile(lv, [10, 50, 90])]
    lum = d["lum_p50"] if np.isfinite(d["lum_p50"]) else 40.0
    thrA = max(130.0, lum + 100.0); thrB = max(200.0, lum + 200.0)
    d["thrA"] = thrA
    nz = None
    for nm, M in (("a", A), ("il", IL), ("v", V)):
        if M.sum() < 5:
            continue
        eM = ndi.binary_erosion(M, B1, iterations=1)
        shell = ndi.binary_dilation(M, B1, iterations=2) & free & ~eM   # wall: outer mask layer + 2 vox outside
        full = ndi.binary_dilation(M, B1, iterations=2) & free           # incl. lumen
        sv = H[shell]
        if sv.size == 0:
            continue
        d[f"{nm}_shell_max"] = float(sv.max()); d[f"{nm}_shell_p99"] = float(np.percentile(sv, 99))
        for t in (130, 200, 300, 400, 600):
            d[f"{nm}_n{t}"] = int((sv > t).sum())
        d[f"{nm}_nA"] = int((sv > thrA).sum()); d[f"{nm}_nB"] = int((sv > thrB).sum())
        d[f"{nm}_massA"] = float(np.clip(sv - thrA, 0, None).sum())
        d[f"{nm}_full_nB"] = int((H[full] > thrB).sum())
        cal = shell & (H > thrA)
        _, ncc = ndi.label(cal); d[f"{nm}_ncompA"] = int(ncc)
        zs = np.where(M.any((1, 2)))[0]
        if len(zs):
            d[f"{nm}_zfracA"] = float(cal[zs].any((1, 2)).mean())
    return d
if __name__ == "__main__":
    os.nice(19)
    Ls = [l.strip() for l in open(ids) if l.strip()]
    res = []
    with Pool(npr) as p:
        for i, r in enumerate(p.imap_unordered(feat, Ls, chunksize=8)):
            if r is not None: res.append(r)
            if i % 1000 == 0: print(i, flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(Ls), flush=True)
