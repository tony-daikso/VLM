"""Bowel-caliber / content / mask-quality features for bowel obstruction, from 3 mm prep HU + unit map
and the TotalSegmentator(total, fast) label map. No embeddings, no labels.
Usage: sbo_feat.py ids.txt "prepdir1,prepdir2" "maskdir1,maskdir2" out.csv nproc   (ids are train/val only)
prep hu/units are [z,y,x] RAS (+y = anterior, so gravity-dependent = lower y in a supine scan);
mask nifti is [x,y,z] -> transpose(2,1,0).
Per region (small bowel unit = TS small_bowel+duodenum, colon, stomach):
  volume, gas/fluid/dense(oral contrast)/soft content, local caliber from the 3D distance transform of the mask
  (diameter = 2 x EDT), volume of the mask lying in segments wider than a threshold (SB >25/30 mm, colon >60 mm,
  stomach >80 mm), fluid fraction of the dilated part, and air-fluid levels (gas voxel directly above a fluid voxel
  along the AP axis, counted as interface voxels and as connected interface patches).
Mask quality (missed dilated loops): inside the deep body (>=15 mm from skin), within the z-range of the bowel units,
voxels NOT covered by any TS label (dilated 1 vox): fluid (0..30 HU, median filtered) and gas (<-400 HU),
total, largest connected component, the part adjacent (<=6 mm) to the small-bowel mask, and the max caliber of
unlabeled fluid blobs (a fluid-filled loop missed by the mask looks like a >=25 mm wide fluid tube).
Also: esophagus high-density voxels (>300 HU) as a crude enteric-tube proxy."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage as nd
ids, preps, mdirs, out, npr = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4], int(sys.argv[5])
B3 = nd.generate_binary_structure(3, 1)
V = 0.027  # ml per 3 mm voxel


def find(dirs, fn):
    for d in dirs:
        p = f"{d}/{fn}"
        if os.path.exists(p): return p
    return None


def bbox(m, pad=2):
    a = np.argwhere(m)
    lo = np.maximum(a.min(0) - pad, 0); hi = np.minimum(a.max(0) + pad + 1, m.shape)
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def region(pre, m, hu, hs, d, thr_mm):
    d[pre + "ml"] = float(m.sum() * V)
    if m.sum() < 100:
        return
    b = bbox(m, 2 + max(int(round(t / 6.0)) for t in thr_mm))   # identical results, smaller arrays
    m, hu, hs = m[b], hu[b], hs[b]
    d[pre + "gas_ml"] = float((m & (hu < -400)).sum() * V)
    d[pre + "fluid_ml"] = float((m & (hs >= -10) & (hs <= 40)).sum() * V)
    d[pre + "dense_ml"] = float((m & (hs > 150)).sum() * V)
    d[pre + "soft_ml"] = float((m & (hs > 40) & (hs <= 150)).sum() * V)
    d[pre + "fluid_frac"] = d[pre + "fluid_ml"] / d[pre + "ml"]
    d[pre + "gas_frac"] = d[pre + "gas_ml"] / d[pre + "ml"]
    e = nd.distance_transform_edt(m) * 3.0          # mm to the mask border
    ev = e[m]
    d[pre + "diam_max"] = float(2 * ev.max())
    d[pre + "diam_p99"] = float(2 * np.percentile(ev, 99))
    d[pre + "diam_p90"] = float(2 * np.percentile(ev, 90))
    for t in thr_mm:
        r = t / 2.0
        core = e >= r
        if core.any():
            it = max(int(round(r / 3.0)), 1)
            dil = nd.binary_dilation(core, B3, iterations=it) & m
            d[f"{pre}dil{t}_ml"] = float(dil.sum() * V)
            d[f"{pre}dil{t}_frac"] = float(dil.sum() / m.sum())
            d[f"{pre}dil{t}_fluidfrac"] = float(((hs >= -10) & (hs <= 40) & dil).sum() / max(dil.sum(), 1))
            lab, n = nd.label(dil, B3)
            d[f"{pre}dil{t}_ncc"] = int(n)
        else:
            d[f"{pre}dil{t}_ml"] = 0.0; d[f"{pre}dil{t}_frac"] = 0.0; d[f"{pre}dil{t}_fluidfrac"] = np.nan; d[f"{pre}dil{t}_ncc"] = 0
    # air-fluid levels: gas voxel with a fluid voxel directly below it (posterior, y-1), both in the region
    gas = m & (hu < -400); fl = m & (hs >= -10) & (hs <= 40)
    itf = np.zeros_like(m); itf[:, 1:, :] = gas[:, 1:, :] & fl[:, :-1, :]
    d[pre + "afl_vox"] = int(itf.sum())
    lab, n = nd.label(itf, nd.generate_binary_structure(3, 2))
    if n:
        sz = np.bincount(lab.ravel())[1:]
        d[pre + "afl_n3"] = int((sz >= 3).sum()); d[pre + "afl_max"] = int(sz.max())
    else:
        d[pre + "afl_n3"] = 0; d[pre + "afl_max"] = 0


def feat(sid):
    pp, mp = find(preps, f"{sid}.npz"), find(mdirs, f"{sid}.nii.gz")
    if pp is None:
        return None
    try:
        z = np.load(pp); hu = z["hu"]; u = z["units"]; names = list(z["unit_names"])
        ts = np.asanyarray(nib.load(mp).dataobj).transpose(2, 1, 0) if mp else None
    except Exception:
        return {"study_id": sid, "err": 1}
    d = {"study_id": sid, "err": 0, "has_ts": int(ts is not None and ts.shape == hu.shape)}
    sb = u == names.index("small_bowel") + 1; co = u == names.index("colon") + 1; st = u == names.index("stomach") + 1
    bw = sb | co | st
    if bw.sum() < 500:
        d["err"] = 2; return d
    a = np.argwhere(bw)
    lo = np.maximum(a.min(0) - 8, 0); hi = np.minimum(a.max(0) + 9, hu.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    H = hu[sl].astype(np.float32); sb, co, st = sb[sl], co[sl], st[sl]
    hs = nd.median_filter(hu[sl], size=3).astype(np.float32)
    region("sb_", sb, H, hs, d, (25, 30, 40))
    region("co_", co, H, hs, d, (60, 80))
    region("st_", st, H, hs, d, (80,))
    if d.get("co_diam_p99") and d.get("sb_diam_p99"):
        d["sb_co_p99_ratio"] = d["sb_diam_p99"] / max(d["co_diam_p99"], 1)
    d["sb_co_vol_ratio"] = d["sb_ml"] / max(d["co_ml"], 1)
    if not d["has_ts"]:
        return d
    T = ts[sl]
    # body and deep body (full-volume body mask, then crop)
    b = hu > -500
    lab, n = nd.label(b, B3)
    if n:
        s = np.bincount(lab.ravel()); s[0] = 0; b = lab == s.argmax()
    b = b[sl]
    for k in range(b.shape[0]): b[k] = nd.binary_fill_holes(b[k])
    deep = nd.binary_erosion(b, iterations=5)
    anyl = nd.binary_dilation(T > 0, B3, iterations=1)
    zr = np.zeros_like(b); zz = np.where((sb | co).any((1, 2)))[0]
    if len(zz): zr[zz[0]:zz[-1] + 1] = True
    unl = deep & ~anyl & zr
    ufl = unl & (hs >= 0) & (hs <= 30)
    ugas = unl & (H < -400)
    sbn = nd.binary_dilation(sb, B3, iterations=2)
    d["unl_fluid_ml"] = float(ufl.sum() * V); d["unl_gas_ml"] = float(ugas.sum() * V)
    d["unl_fluid_adjsb_ml"] = float((ufl & sbn).sum() * V); d["unl_gas_adjsb_ml"] = float((ugas & sbn).sum() * V)
    lab, n = nd.label(ufl, B3)
    if n:
        s = np.bincount(lab.ravel()); s[0] = 0
        d["unl_fluid_maxcc_ml"] = float(s.max() * V)
        bb = bbox(ufl); e = np.zeros(ufl.shape, np.float64); e[bb] = nd.distance_transform_edt(ufl[bb]) * 3.0
        d["unl_fluid_diam_max"] = float(2 * e.max())
        d["unl_fluid_ge25_ml"] = float((e >= 12.5).sum() * V)
    else:
        d["unl_fluid_maxcc_ml"] = 0.0; d["unl_fluid_diam_max"] = 0.0; d["unl_fluid_ge25_ml"] = 0.0
    # gas+fluid mixture outside labels (missed gas-containing loop): unlabeled gas within 6 mm of unlabeled fluid
    d["unl_gasnearfluid_ml"] = float((ugas & nd.binary_dilation(ufl, B3, iterations=2)).sum() * V)
    es = T == 15
    d["esoph_hi_vox"] = int((es & (H > 300)).sum())
    return d


if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    res = []
    if os.path.exists(out):
        old = pd.read_csv(out); res = old.to_dict("records"); done = set(old.study_id); L = [s for s in L if s not in done]
        print("resume: done", len(done), "todo", len(L), flush=True)
    with Pool(npr) as p:
        for i, r in enumerate(p.imap_unordered(feat, L, chunksize=4)):
            if r is not None: res.append(r)
            if (i + 1) % 500 == 0:
                pd.DataFrame(res).to_csv(out, index=False); print("progress", i + 1, "of", len(L), flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), flush=True)
