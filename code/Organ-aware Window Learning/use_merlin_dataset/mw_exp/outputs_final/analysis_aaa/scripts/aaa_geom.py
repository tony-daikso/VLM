"""Aortic geometry + intra/peri-aortic HU features from prep npz (3 mm HU) + TS fast multilabel mask (same grid).
Usage: aaa_geom.py ids.txt prep_dir mask_dir out.csv nproc
Volumes are (z,y,x) after canonical RAS + transpose, so z index increases cranially (checked per case via lungs/kidneys).
Per axial slice of the aorta mask (largest 2D component): equivalent diameter 2*sqrt(area/pi), minor/major axis of
the moment ellipse (4*sqrt(eig)), max Feret extent. Segments: infrarenal = below the renal level (mean z of kidney
centroids; fallback L1/L2 boundary) down to the lowest aortic slice (bifurcation); suprarenal-abdominal = renal level
to top of kidneys; thoracic = above. Robust max = max of a 3-slice running median."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
AO, ILL, ILR, KR, KL, L1, L2, LUNG = 52, 65, 66, 2, 3, 31, 30, (10, 11, 12, 13, 14)
S8 = np.ones((3, 3), bool)
B1 = ndi.generate_binary_structure(3, 1)

def slice_stats(M2, px):
    lab, n = ndi.label(M2, S8)
    if n == 0:
        return None
    sz = ndi.sum(M2, lab, range(1, n + 1)); k = int(np.argmax(sz)) + 1
    C = lab == k; a = C.sum()
    yx = np.argwhere(C).astype(np.float32)
    eq = 2 * np.sqrt(a * px * px / np.pi)
    if a >= 3:
        cov = np.cov(yx.T) + np.eye(2) / 12.0
        ev = np.clip(np.linalg.eigvalsh(cov), 1e-6, None)
        mn, mj = 4 * np.sqrt(ev[0]) * px, 4 * np.sqrt(ev[1]) * px
    else:
        mn = mj = eq
    if a >= 2:
        er = C & ~ndi.binary_erosion(C, S8); pts = np.argwhere(er)
        fe = (pdist(pts).max() + 1) * px if len(pts) >= 2 else px
    else:
        fe = px
    return eq, mn, mj, fe, n

def rmax(v):
    v = np.asarray(v, float)
    if len(v) == 0: return np.nan
    if len(v) < 3: return float(np.nanmin(v))
    return float(np.max([np.median(v[i:i + 3]) for i in range(len(v) - 2)]))

def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz"); hu = z["hu"]; aff = z["affine"]
        lab = np.asanyarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).astype(np.uint8).transpose(2, 1, 0)
    except Exception:
        return None
    if lab.shape != hu.shape:
        return {"study_id": sid, "grid_mismatch": 1}
    px = float(np.abs(aff[0, 0])) or 3.0
    pz = float(np.abs(aff[2, 2])) or 3.0
    d = {"study_id": sid, "grid_mismatch": 0, "nz": lab.shape[0], "px": px}
    A = lab == AO
    zs = np.where(A.any((1, 2)))[0]
    d["a_vox"] = int(A.sum())
    if len(zs) < 3:
        return d
    kid = np.isin(lab, [KR, KL]); kz = np.where(kid.any((1, 2)))[0]
    lz = np.where(np.isin(lab, LUNG).any((1, 2)))[0]
    d["z_up_ok"] = int(len(kz) > 0 and len(lz) > 0 and lz.mean() > kz.mean()) if len(lz) and len(kz) else -1
    if len(kz):
        cz = [ndi.center_of_mass(lab == k)[0] for k in (KR, KL) if (lab == k).any()]
        zren = float(np.mean(cz)); ktop = int(kz.max())
    else:
        l1 = np.where((lab == L1).any((1, 2)))[0]; l2 = np.where((lab == L2).any((1, 2)))[0]
        zren = float(l1.min()) if len(l1) else (float(l2.max()) if len(l2) else np.nan); ktop = np.nan
    d["z_ren"] = zren; d["a_zmin"] = int(zs.min()); d["a_zmax"] = int(zs.max())
    d["a_top_at_fov"] = int(zs.max() >= lab.shape[0] - 2); d["a_bot_at_fov"] = int(zs.min() <= 1)
    d["has_kidney"] = int(len(kz) > 0)
    d["il_vox"] = int(np.isin(lab, [ILL, ILR]).sum())
    per = {}
    for zz in zs:
        s = slice_stats(A[zz], px)
        if s: per[zz] = s
    zk = np.array(sorted(per)); P = np.array([per[k][:4] for k in zk])  # eq, minor, major, feret
    ncomp = np.array([per[k][4] for k in zk])
    segs = {"all": np.ones(len(zk), bool)}
    if np.isfinite(zren):
        segs["infra"] = zk < zren
        segs["abd"] = zk < (ktop + 2 if np.isfinite(ktop) else zren + 15)
        segs["thor"] = zk >= (ktop + 2 if np.isfinite(ktop) else zren + 15)
    for nm, m in segs.items():
        d[f"{nm}_nsl"] = int(m.sum())
        if m.sum() == 0:
            continue
        for j, q in enumerate(["eq", "minor", "major", "feret"]):
            d[f"{nm}_{q}_max"] = float(P[m, j].max()); d[f"{nm}_{q}_rmax"] = rmax(P[m, j])
        d[f"{nm}_eq_med"] = float(np.median(P[m, 0]))
        d[f"{nm}_vol_ml"] = float(sum(A[k].sum() for k in zk[m]) * px * px * pz / 1000)
    if "infra" in segs and segs["infra"].sum() and segs["abd"].sum():
        ab = segs["abd"] & ~segs["infra"]
        ref = np.median(P[ab, 0]) if ab.sum() >= 2 else np.median(P[segs["abd"], 0])
        d["infra_ratio"] = d["infra_eq_rmax"] / ref if ref > 0 else np.nan
        d["infra_len_mm"] = float(segs["infra"].sum() * pz)
        d["infra_multicomp_frac"] = float((ncomp[segs["infra"]] > 1).mean())
    # HU inside the aorta mask: lumen reference (suprarenal / thoracic core) vs infrarenal core
    core = ndi.binary_erosion(A, B1)
    if core.sum() < 20: core = A
    Zi = np.zeros(lab.shape[0], bool); Zr = np.zeros(lab.shape[0], bool)
    if np.isfinite(zren):
        Zi[:max(int(zren), 0)] = True
        Zr[max(int(zren), 0):(int(ktop) + 3 if np.isfinite(ktop) else int(zren) + 15)] = True
    else:
        Zr[:] = True
    ref = hu[core & Zr[:, None, None]]
    if ref.size < 20: ref = hu[core]
    lum = float(np.median(ref)); d["lum_ref"] = lum
    ai = A & Zi[:, None, None]
    if ai.sum() >= 10:
        v = hu[ai].astype(np.float32)
        d["infra_hu_med"] = float(np.median(v)); d["infra_hu_p10"] = float(np.percentile(v, 10))
        d["infra_lowfrac"] = float((v < lum - 80).mean()) if lum > 120 else np.nan   # thrombus inside mask (contrast only)
        d["infra_low30_80"] = float(((v > 0) & (v < 80)).mean())
        # peri-aortic soft-tissue cuff: unlabeled voxels 0..90 HU within 4 dilations (12 mm) connected to the aorta
        idx = np.argwhere(ai); lo = np.maximum(idx.min(0) - 6, 0); hi = idx.max(0) + 7
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        Ai = ai[sl]; L = lab[sl]; H = hu[sl]
        cand = (L == 0) & (H > 0) & (H < 90)
        G = Ai.copy()
        for _ in range(4):
            G = G | (ndi.binary_dilation(G, generate := np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]], bool)) & cand)
        eqg = []
        for k in range(G.shape[0]):
            if Ai[k].any():
                s = slice_stats(G[k], px)
                if s: eqg.append(s[0])
        d["infra_grow_eq_rmax"] = rmax(eqg)
        d["infra_grow_ratio"] = d["infra_grow_eq_rmax"] / d["infra_eq_rmax"] if d.get("infra_eq_rmax") else np.nan
    # high-HU (metal / stent graft vs calcification) inside / around aorta + iliacs
    V = np.isin(lab, [AO, ILL, ILR])
    VV = ndi.binary_dilation(V, B1, iterations=2)
    hv = hu[VV]
    for t in (800, 1200, 1600, 2000):
        d[f"v_n{t}"] = int((hv > t).sum())
    d["v_max"] = float(hv.max()) if hv.size else np.nan
    d["inV_n800"] = int((hu[V] > 800).sum())
    # iliac max diameters
    for nm, k in (("ilL", ILL), ("ilR", ILR)):
        M = lab == k; zz = np.where(M.any((1, 2)))[0]
        eq = [slice_stats(M[q], px)[0] for q in zz]
        d[f"{nm}_eq_rmax"] = rmax(eq) if len(eq) else np.nan
        d[f"{nm}_eq_max"] = float(max(eq)) if len(eq) else np.nan
    d["il_eq_rmax"] = np.nanmax([d["ilL_eq_rmax"], d["ilR_eq_rmax"]]) if np.isfinite([d["ilL_eq_rmax"], d["ilR_eq_rmax"]]).any() else np.nan
    return d

if __name__ == "__main__":
    ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    os.nice(19)
    Ls = [l.strip() for l in open(ids) if l.strip()]
    res = []
    with Pool(npr) as p:
        for i, r in enumerate(p.imap_unordered(feat, Ls, chunksize=8)):
            if r is not None: res.append(r)
            if i % 1000 == 0:
                print(i, flush=True)
                pd.DataFrame(res).to_csv(out + ".part", index=False)
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(Ls), flush=True)
else:
    prep = mdir = None
