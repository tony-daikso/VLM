"""Local liver HU features for 局部脂肪浸潤 (focal fatty infiltration), from prep npz (3 mm HU, z-y-x, units)
+ TotalSegmentator total masks (x-y-z). CPU only.

Features (liver eroded 2 vox; HU smoothed with a 3x3x3 median before thresholding):
  * heterogeneity: std, MAD, IQR, skew, p1/p5/p10/p25 - median
  * hypodense volume (mL) / fraction: smoothed HU < median-15 / -25 / -40; largest connected hypodense blob (mL)
  * typical locations (also volume of hypodense voxels, region median - liver median):
      gbf  = liver within 15 mm of the gallbladder (GB fossa)
      porta= liver within 15 mm of the portal vein (TS id 64)
      falc = anterior paramedian liver: |x - spine midline| <= 30 mm and anterior half of the liver on that slice
             (falciform ligament / ligamentum teres fissure / segment IV anterior)
  * max pooling over local blocks (5x5x5 vox = 15 mm cubes with >=60 % liver): min / p5 block median - liver median,
    max block fraction < median-25, std of block medians
Usage: hs_focal_hu.py ids.txt prep_dir mask_dir out.csv nproc     (ids must already be train/val only)
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage, stats

TS_PORTAL = 64
VOX_ML = 0.027  # 3 mm isotropic


def rel(d, pre, v, med):
    d[pre + "_n"] = int(v.size)
    if v.size < 20:
        return
    d[pre + "_dmed"] = float(np.median(v) - med)
    for t in (15, 25):
        h = v < med - t
        d[f"{pre}_ml_lt{t}"] = float(h.sum() * VOX_ML); d[f"{pre}_frac_lt{t}"] = float(h.mean())


def feat(sid):
    ids_, prep, mdir = ARGS
    try:
        z = np.load(f"{prep}/{sid}.npz")
        hu = z["hu"].astype(np.float32); u = z["units"]; names = list(z["unit_names"])
    except Exception:
        return None
    d = {"study_id": sid}
    lv = u == names.index("liver") + 1
    if lv.sum() < 500:
        d["fl_err"] = "noliver"; return d
    le = ndimage.binary_erosion(lv, iterations=2)
    if le.sum() < 300:
        le = lv
    zz, yy, xx = np.where(lv)
    z0, z1, y0, y1, x0, x1 = zz.min(), zz.max() + 1, yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    H = hu[sl]; LE = le[sl]
    Hs = ndimage.median_filter(np.clip(H, -200, 400), size=3)
    v = H[LE]; vs = Hs[LE]; med = float(np.median(v))
    d["fl_med"] = med; d["fl_std"] = float(v.std()); d["fl_mad"] = float(np.median(np.abs(v - med)))
    d["fl_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))); d["fl_skew"] = float(stats.skew(v))
    for p, x in zip((1, 5, 10, 25), np.percentile(v, (1, 5, 10, 25))):
        d[f"fl_p{p}_rel"] = float(x - med)
    for p, x in zip((1, 5, 10), np.percentile(vs, (1, 5, 10))):
        d[f"fl_sm_p{p}_rel"] = float(x - med)
    d["fl_sm_std"] = float(vs.std())
    for t in (15, 25, 40):
        h = (Hs < med - t) & LE
        d[f"fl_ml_lt{t}"] = float(h.sum() * VOX_ML); d[f"fl_frac_lt{t}"] = float(h.sum() / LE.sum())
        if t in (15, 25):
            lab, n = ndimage.label(h)
            d[f"fl_blob_lt{t}_ml"] = float(np.bincount(lab.ravel())[1:].max() * VOX_ML) if n else 0.0
            d[f"fl_nblob_lt{t}"] = int((np.bincount(lab.ravel())[1:] * VOX_ML >= 1.0).sum()) if n else 0
    # --- locations
    gb = u[sl] == names.index("gallbladder") + 1
    if gb.sum() >= 30:
        r = ndimage.binary_dilation(gb, iterations=5) & LE
        rel(d, "gbf", Hs[r], med)
    else:
        d["gbf_n"] = 0
    try:
        m = np.asarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
        if m.shape != hu.shape:
            raise ValueError("shape")
        pv = m[sl] == TS_PORTAL
        if pv.sum() >= 10:
            r = ndimage.binary_dilation(pv, iterations=5) & LE
            rel(d, "porta", Hs[r], med)
        else:
            d["porta_n"] = 0
    except Exception as e:
        d["fl_mask_err"] = str(e)[:30]
    sp = u == names.index("lumbar_spine") + 1
    ao = u == names.index("aorta") + 1
    if sp.sum() >= 50 and ao.sum() >= 50:
        sx = np.median(np.where(sp)[2]); sy = np.median(np.where(sp)[1]); ay = np.median(np.where(ao)[1])
        s = 1.0 if ay > sy else -1.0                        # +1: anterior = increasing y
        Z, Y, X = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        ymid = np.full(H.shape[0], np.nan)
        for k in range(H.shape[0]):
            ys = np.where(LE[k].any(1))[0]
            if len(ys):
                ymid[k] = (ys.min() + ys.max()) / 2 + y0
        ant = s * (Y - ymid[:, None, None]) > 0
        r = LE & ant & (np.abs(X - sx) <= 10)
        rel(d, "falc", Hs[r], med)
    else:
        d["falc_n"] = 0
    # --- max pooling over 15 mm blocks
    B = 5; bm, bf = [], []
    for a in range(0, H.shape[0] - B + 1, B):
        for b in range(0, H.shape[1] - B + 1, B):
            for c in range(0, H.shape[2] - B + 1, B):
                mk = LE[a:a + B, b:b + B, c:c + B]
                if mk.mean() >= 0.6:
                    hb = Hs[a:a + B, b:b + B, c:c + B][mk]
                    bm.append(np.median(hb)); bf.append((hb < med - 25).mean())
    d["blk_n"] = len(bm)
    if len(bm) >= 5:
        bm = np.array(bm) - med; bf = np.array(bf)
        d["blk_min_rel"] = float(bm.min()); d["blk_p5_rel"] = float(np.percentile(bm, 5))
        d["blk_max_rel"] = float(bm.max()); d["blk_std"] = float(bm.std())
        d["blk_max_frac25"] = float(bf.max()); d["blk_n_lt15"] = int((bm < -15).sum())
    return d


def init(a):
    global ARGS
    ARGS = a


if __name__ == "__main__":
    ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    L = [l.strip() for l in open(ids) if l.strip()]
    if len(sys.argv) > 6:
        L = L[:int(sys.argv[6])]
    with Pool(npr, initializer=init, initargs=((ids, prep, mdir),)) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), flush=True)
