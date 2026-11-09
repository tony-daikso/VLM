"""RLQ / cecal-region handcrafted HU features from prep npz (3 mm HU + unit label map).
Usage: ap_hu.py ids.txt prep_dir out.csv nproc
Orientation determined per case: axis with largest liver-vs-spleen centroid offset = L-R (liver side = patient right);
superior axis from liver vs lumbar_spine. Cecum proxy = most caudal 45 mm of colon voxels lying >=30 mm right of the
spine midline. ROI shells (outside any TS unit, inside body) at <=15 mm and <=30 mm from the cecal proxy.
Same features in the mirrored LLQ (left colon, same z band) as a within-patient reference."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage
ids, prep, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def cent(u, k):
    a = np.argwhere(u == k)
    return a.mean(0) if len(a) > 50 else None

def region_feats(hu, u, seed, pre, d):
    """seed: bool mask (cecal proxy). Features in a crop around it."""
    a = np.argwhere(seed)
    lo = np.maximum(a.min(0) - 12, 0); hi = np.minimum(a.max(0) + 13, hu.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    H = hu[sl].astype(np.float32); U = u[sl]; S = seed[sl]
    dist = ndimage.distance_transform_edt(~S) * 3.0          # mm
    body = H > -400
    outside = (U == 0) & body
    d[pre + "seed_vox"] = int(S.sum())
    sv = H[S]
    d[pre + "wall_p50"] = float(np.median(sv)); d[pre + "wall_p90"] = float(np.percentile(sv, 90))
    d[pre + "lumen_gas_frac"] = float((sv < -400).mean())
    d[pre + "lumen_hi_frac"] = float((sv > 200).mean())       # oral contrast / stool / colith inside cecum
    for r in (15, 30):
        R = outside & (dist > 0) & (dist <= r)
        v = H[R]
        if v.size < 20: continue
        fat = v[(v >= -190) & (v <= -30)]
        soft = v[(v >= -190) & (v <= 60)]
        d[f"{pre}r{r}_n"] = int(v.size)
        d[f"{pre}r{r}_fat_med"] = float(np.median(fat)) if fat.size > 10 else np.nan
        d[f"{pre}r{r}_fat_p75"] = float(np.percentile(fat, 75)) if fat.size > 10 else np.nan
        d[f"{pre}r{r}_fat_frac"] = float(fat.size / v.size)
        d[f"{pre}r{r}_strand_frac"] = float(((soft >= -70) & (soft <= 20)).mean()) if soft.size else np.nan   # hazy fat + fluid among non-gas
        d[f"{pre}r{r}_fluid_frac"] = float(((v >= 0) & (v <= 30)).mean())
        d[f"{pre}r{r}_soft_frac"] = float(((v > 30) & (v <= 100)).mean())
        d[f"{pre}r{r}_gas_frac"] = float((v < -400).mean())
        d[f"{pre}r{r}_med"] = float(np.median(v))
        # appendicolith / dense focus: small connected components > 200 HU outside bowel units, not bone-sized
        hi = R & (H > 200)
        lab, n = ndimage.label(hi)
        if n:
            sz = np.bincount(lab.ravel())[1:]
            small = sz[(sz >= 1) & (sz <= 15)]
            d[f"{pre}r{r}_hi_small_n"] = int(len(small)); d[f"{pre}r{r}_hi_small_vox"] = int(small.sum())
            d[f"{pre}r{r}_hi_big_vox"] = int(sz[sz > 15].sum())
        else:
            d[f"{pre}r{r}_hi_small_n"] = 0; d[f"{pre}r{r}_hi_small_vox"] = 0; d[f"{pre}r{r}_hi_big_vox"] = 0
    return d

def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz")
    except Exception:
        return None
    names = list(z["unit_names"]); hu = z["hu"]; u = z["units"]
    d = {"study_id": sid}
    ci = names.index("colon") + 1; bi = names.index("small_bowel") + 1
    col = u == ci
    d["colon_vox"] = int(col.sum()); d["sb_vox"] = int((u == bi).sum())
    cl, cs, cp = cent(u, names.index("liver") + 1), cent(u, names.index("spleen") + 1), cent(u, names.index("lumbar_spine") + 1)
    if cl is None or cp is None or d["colon_vox"] < 500:
        d["ok"] = 0; return d
    if cs is None:
        ck = cent(u, names.index("kidney_left") + 1); cs = ck if ck is not None else cp
    off = cl - cs; lr = int(np.argmax(np.abs(off[1:]))) + 1 if abs(off[0]) < max(abs(off[1:])) else 2
    sgn_r = 1 if cl[lr] > cs[lr] else -1
    si = 0; sgn_s = 1 if cl[si] > cp[si] else -1
    mid = cp[lr]
    # global fat reference: fat outside units in whole body
    body = hu > -400
    g = hu[(u == 0) & body]
    gf = g[(g >= -190) & (g <= -30)]
    d["glob_fat_med"] = float(np.median(gf)) if gf.size else np.nan
    d["glob_fluid_frac"] = float(((g >= 0) & (g <= 30)).mean()) if g.size else np.nan
    coords = np.argwhere(col)
    side = (coords[:, lr] - mid) * sgn_r * 3.0            # mm to patient right
    zz = coords[:, si] * sgn_s                            # larger = more superior
    d["ok"] = 1
    for pre, sel in (("rlq_", side >= 30), ("llq_", side <= -30)):
        c = coords[sel]
        if len(c) < 50: continue
        zc = zz[sel]; zmin = np.percentile(zc, 1)
        if pre == "llq_" and "rlq_zmin" in d: zmin = d["rlq_zmin"]   # mirror at the same craniocaudal band
        band = c[(zc >= zmin) & (zc <= zmin + 15)]
        if len(band) < 20: continue
        if pre == "rlq_": d["rlq_zmin"] = zmin
        seed = np.zeros(u.shape, bool); seed[tuple(band.T)] = True
        # caudal tip height relative to spine centroid (mm) - cecal position
        d[pre + "z_rel_spine_mm"] = float((zmin - cp[si] * sgn_s) * 3.0)
        region_feats(hu, u, seed, pre, d)
    for r in (15, 30):
        for f in ("fat_med", "strand_frac", "fluid_frac", "soft_frac", "hi_small_n"):
            a, bb = d.get(f"rlq_r{r}_{f}"), d.get(f"llq_r{r}_{f}")
            if a is not None and bb is not None: d[f"dlr_r{r}_{f}"] = a - bb
        if f"rlq_r{r}_fat_med" in d: d[f"dglob_r{r}_fat_med"] = d[f"rlq_r{r}_fat_med"] - d["glob_fat_med"]
    return d

if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L))
