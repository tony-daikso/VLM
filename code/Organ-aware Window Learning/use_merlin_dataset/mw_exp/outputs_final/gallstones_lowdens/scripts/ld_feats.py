"""Low-density (hypodense / fat / gas stone) GB features. Train/val ids only (ids list pre-filtered by split). Read-only on sources.
mode 3mm : prep npz (3 mm HU + TS unit map; arrays z,y,x; affine diag)  -> prefix ld_
mode 15  : gs15 1.5 mm crops (hu 96^3, packed gbmask; +y anterior)      -> prefix ld15_
Lumen = GB mask eroded 1 voxel (fallback: full mask if < 10 vox). bile median = lumen median.
Usage: ld_feats.py MODE ids.txt SRC_DIR out.csv nproc"""
import sys, os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage
MODE, IDS, SRC, OUT, NP = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
ST = ndimage.generate_binary_structure(3, 1)

def surface_mm2(m, sp):
    # voxel-face surface area (z,y,x spacing sp)
    a = 0.0; areas = (sp[1] * sp[2], sp[0] * sp[2], sp[0] * sp[1])
    for ax in range(3):
        p = np.pad(m, [(1, 1) if i == ax else (0, 0) for i in range(3)])
        a += np.count_nonzero(np.diff(p.astype(np.int8), axis=ax)) * areas[ax]
    return a

def feats(hu, gb, sp, yworld):
    """hu,gb arrays (z,y,x); sp spacing (z,y,x) mm; yworld: 1-D world-y (anterior +) per y index."""
    d = {}; vv = float(np.prod(sp)) / 1000.0
    d["gb_ml"] = gb.sum() * vv
    if gb.sum() < 10: return d
    lum = ndimage.binary_erosion(gb, ST, 1)
    if lum.sum() < 10: lum = gb
    v = hu[lum].astype(np.float32); med = float(np.median(v))
    d["lum_ml"] = lum.sum() * vv; d["bile_med"] = med; d["lum_min"] = float(v.min())
    for p in (1, 5, 10): d[f"lum_p{p}"] = float(np.percentile(v, p))
    for p in (1, 5, 10): d[f"lum_p{p}_rel"] = d[f"lum_p{p}"] - med
    d["lum_min_rel"] = d["lum_min"] - med
    for k in (10, 20, 30): d[f"frac_lt_med{k}"] = float((v < med - k).mean())
    d["frac_lt0"] = float((v < 0).mean()); d["frac_ltm100"] = float((v < -100).mean())
    d["n_lt0"] = int((v < 0).sum()); d["n_ltm100"] = int((v < -100).sum())
    d["lum_std"] = float(v.std()); d["lum_iqr"] = float(np.subtract(*np.percentile(v, [75, 25])))
    h, _ = np.histogram(np.clip(v, -200, 300), bins=np.arange(-200, 305, 5)); p = h[h > 0] / h.sum()
    d["lum_entropy"] = float(-(p * np.log2(p)).sum())
    # robust (MAD) heterogeneity
    d["lum_mad"] = float(np.median(np.abs(v - med)))
    # largest hypodense connected component (lumen & HU < med-20), after 1-vox smoothing of HU to suppress noise
    hs = ndimage.uniform_filter(hu.astype(np.float32), 3) if MODE == "15" else hu.astype(np.float32)
    for k in (20,):
        lab, n = ndimage.label(lum & (hs < med - k), ST)
        if n:
            sz = np.bincount(lab.ravel())[1:]; i = int(sz.argmax()) + 1; comp = lab == i
            V = comp.sum() * np.prod(sp); A = surface_mm2(comp, sp)
            d[f"hypo{k}_maxcc_ml"] = V / 1000.0; d[f"hypo{k}_ncc"] = int(n)
            d[f"hypo{k}_maxcc_frac"] = float(comp.sum() / lum.sum())
            d[f"hypo{k}_sphericity"] = float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / A) if A > 0 else np.nan
            d[f"hypo{k}_maxcc_meanrel"] = float(hu[comp].mean() - med)
        else:
            d[f"hypo{k}_maxcc_ml"] = 0.0; d[f"hypo{k}_ncc"] = 0; d[f"hypo{k}_maxcc_frac"] = 0.0
    # fat-like CC (< 0 HU) largest
    lab, n = ndimage.label(lum & (hs < 0), ST)
    d["fat_maxcc_ml"] = float(np.bincount(lab.ravel())[1:].max() * vv) if n else 0.0
    # dependent (posterior third, lowest world y) vs nondependent (anterior third), over full GB mask
    yi = np.nonzero(gb)[1]; yw = yworld[yi]; q1, q2 = np.percentile(yw, [33.3, 66.7])
    hv = hu[gb].astype(np.float32)
    dep, nde = hv[yw <= q1], hv[yw >= q2]
    if dep.size >= 5 and nde.size >= 5:
        d["dep_minus_nondep_med"] = float(np.median(dep) - np.median(nde))
        d["dep_minus_nondep_p10"] = float(np.percentile(dep, 10) - np.percentile(nde, 10))
        d["dep_minus_nondep_p90"] = float(np.percentile(dep, 90) - np.percentile(nde, 90))
    return d

def run(sid):
    try:
        if MODE == "3mm":
            z = np.load(f"{SRC}/{sid}.npz"); names = list(z["unit_names"]); A = z["affine"]
            gb = z["units"] == names.index("gallbladder") + 1; hu = z["hu"]
            sp = np.abs([A[2, 2], A[1, 1], A[0, 0]]); yworld = A[1, 1] * np.arange(hu.shape[1]) + A[1, 3]
            pre = "ld_"
        else:
            z = np.load(f"{SRC}/{sid}.npz")
            if not bool(z["present"][0]): return dict(study_id=sid)
            hu = z["hu"][0]; gb = np.unpackbits(z["gbmask"])[:hu.size].reshape(hu.shape).astype(bool)
            sp = np.array([1.5, 1.5, 1.5]); yworld = 1.5 * np.arange(hu.shape[1])  # +y = anterior
            pre = "ld15_"
        return {"study_id": sid, **{pre + k: v for k, v in feats(hu, gb, sp, yworld).items()}}
    except FileNotFoundError:
        return None
    except Exception as e:
        return dict(study_id=sid, err=str(e)[:200])

if __name__ == "__main__":
    L = [l.strip() for l in open(IDS) if l.strip()]
    if MODE == "15": L = [s for s in L if os.path.exists(f"{SRC}/{s}.npz")]
    with Pool(NP) as p:
        R = [r for r in p.imap_unordered(run, L, chunksize=16) if r is not None]
    pd.DataFrame(R).to_csv(OUT, index=False); print("done", len(R), "of", len(L), flush=True)
