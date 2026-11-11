"""M4 extension: deterministic per-unit geometry / HU / slice-range table from the 3 mm TotalSegmentator 'total'
label map (masks/{id}.nii.gz, RAS, 3 mm) + 3 mm prep HU (prep/{id}.npz 'hu', z,y,x).  CPU only, no fitting.

Units: kidney_l (TS 3), kidney_r (2), liver (5), spleen (1), aorta (52), vertebrae T8..L5 (36..27), sacrum (25).
Per unit u:
  u_vox, u_ml            voxel count, volume (ml, from affine)
  u_zlo, u_zhi           slice index range in the 3 mm grid (k=0 = most INFERIOR slice, RAS)
  u_zlo_mm, u_zhi_mm     same range as world z (mm, scanner coordinates)
  u_hu_mean, u_hu_med    mean / median HU inside the mask (vertebrae: whole vertebra incl. cortex + posterior elements)
  u_cut                  1 if the unit touches the first/last slice (truncated by the FOV)
Kidneys also: u_hu_p10, u_lowhu_ml (-10..25 HU voxels inside the mask, ml), u_lowhu_frac, u_cyst_ml (TS kidney_cyst).
Vertebrae also: u_zext (n slices), u_ml_nb (volume / mean volume of valid neighbours), u_zext_nb (same for z extent).
Case: n_z, z0_mm (world z of k=0), dz_mm.
usage: python m4ext_units.py ids.txt out_root out.csv nproc     (out_root has prep/ and masks/)"""
import sys, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool

UNITS = {"kidney_l": 3, "kidney_r": 2, "liver": 5, "spleen": 1, "aorta": 52, "sacrum": 25}
VERT = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
VLAB = {v: l for v, l in zip(VERT, [36, 35, 34, 33, 32, 31, 30, 29, 28, 27])}
ALL = {**UNITS, **VLAB}
CYST = {"kidney_l": 23, "kidney_r": 24}
MINVOX = 20
ROOT = None


def feat(sid):
    try:
        z = np.load(f"{ROOT}/prep/{sid}.npz")
        hu = z["hu"]
        img = nib.load(f"{ROOT}/masks/{sid}.nii.gz")
        lab = np.asarray(img.dataobj).transpose(2, 1, 0)
    except Exception:
        return None
    if lab.shape != hu.shape:
        return None
    A = img.affine
    nz = lab.shape[0]
    vml = abs(np.linalg.det(A[:3, :3])) / 1000.0
    d = {"study_id": sid, "n_z": nz, "z0_mm": float(A[2, 3]), "dz_mm": float(A[2, 2])}
    lab_i = lab.astype(np.int32)
    flat = lab_i.ravel(); huf = hu.ravel().astype(np.float64)
    cnt = np.bincount(flat, minlength=256)
    hsum = np.bincount(flat, weights=huf, minlength=256)
    perz = np.bincount((np.arange(nz, dtype=np.int32)[:, None, None] * 256 + lab_i).ravel(),
                       minlength=nz * 256).reshape(nz, 256)
    for u, k in ALL.items():
        n = int(cnt[k]); d[f"{u}_vox"] = n
        if n < MINVOX:
            continue
        zs = np.where(perz[:, k] > 0)[0]
        lo, hi = int(zs.min()), int(zs.max())
        d[f"{u}_ml"] = n * vml
        d[f"{u}_zlo"], d[f"{u}_zhi"] = lo, hi
        d[f"{u}_zlo_mm"] = float(A[2, 2] * lo + A[2, 3]); d[f"{u}_zhi_mm"] = float(A[2, 2] * hi + A[2, 3])
        d[f"{u}_cut"] = int(lo == 0 or hi == nz - 1)
        d[f"{u}_hu_mean"] = float(hsum[k] / n)
        m = lab == k
        v = hu[m].astype(np.float32)
        d[f"{u}_hu_med"] = float(np.median(v))
        if u in CYST:
            d[f"{u}_hu_p10"] = float(np.percentile(v, 10))
            lw = int(((v >= -10) & (v <= 25)).sum())
            d[f"{u}_lowhu_ml"] = lw * vml; d[f"{u}_lowhu_frac"] = lw / n
            d[f"{u}_cyst_ml"] = int(cnt[CYST[u]]) * vml
        if u in VLAB:
            d[f"{u}_zext"] = hi - lo + 1
    for i, v in enumerate(VERT):
        if f"{v}_ml" not in d or d[f"{v}_cut"]:
            continue
        nb = [VERT[j] for j in (i - 1, i + 1) if 0 <= j < len(VERT) and f"{VERT[j]}_ml" in d and not d[f"{VERT[j]}_cut"]]
        if nb:
            d[f"{v}_ml_nb"] = d[f"{v}_ml"] / np.mean([d[f"{b}_ml"] for b in nb])
            d[f"{v}_zext_nb"] = d[f"{v}_zext"] / np.mean([d[f"{b}_zext"] for b in nb])
    return d


if __name__ == "__main__":
    ids, ROOT, out, npr = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], int(sys.argv[4])
    L = [l.strip() for l in open(ids) if l.strip()]
    res, miss = [], 0
    with Pool(npr) as p:
        for i, r in enumerate(p.imap_unordered(feat, L, chunksize=8)):
            if r is None:
                miss += 1
            else:
                res.append(r)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(L)} miss={miss}", flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), "miss", miss, flush=True)
