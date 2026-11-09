"""Steatosis HU / phase / fat features from prep npz (3mm HU, z-y-x) + TS total masks (3mm, x-y-z).
Usage: hs_hu.py ids.txt prep_dir mask_dir out.csv nproc   (ids must already be train/val only)"""
import sys, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage
ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
# TotalSegmentator v2 'total' class ids
TS = {"spleen": 1, "kidney_right": 2, "kidney_left": 3, "liver": 5, "aorta": 52, "ivc": 63, "portal": 64, "L1": 31, "L3": 29}


def med(hu, m, n=30):
    return float(np.median(hu[m])) if m.sum() >= n else np.nan


def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz")
        hu = z["hu"].astype(np.float32); u = z["units"]; names = list(z["unit_names"])
    except Exception:
        return None
    d = {"study_id": sid}
    lv = u == names.index("liver") + 1
    d["liver_vox"] = int(lv.sum())
    d["n_slices"] = hu.shape[0]
    if lv.sum() >= 100:
        zz = np.where(lv.any((1, 2)))[0]
        d["liver_z0"], d["liver_z1"] = int(zz[0]), int(zz[-1])
        d["liver_cut"] = int(zz[0] == 0 or zz[-1] == hu.shape[0] - 1)
        v = hu[lv]
        d["liver_mean"] = float(v.mean()); d["liver_std"] = float(v.std())
        for p, x in zip([5, 10, 25, 50, 75, 90, 95], np.percentile(v, [5, 10, 25, 50, 75, 90, 95])):
            d[f"liver_p{p}"] = float(x)
        le = ndimage.binary_erosion(lv, iterations=2)
        d["liver_er_med"] = med(hu, le)
        if le.sum() >= 30:
            ve = hu[le]
            d["liver_er_iqr"] = float(np.subtract(*np.percentile(ve, [75, 25])))
            d["liver_frac_lt40"] = float((ve < 40).mean()); d["liver_frac_lt0"] = float((ve < 0).mean())
            # heterogeneity: spread of per-slab (5 slices) medians
            zs = np.where(le.any((1, 2)))[0]
            slab = [np.median(hu[a:a + 5][le[a:a + 5]]) for a in range(zs[0], zs[-1] + 1, 5) if le[a:a + 5].sum() > 50]
            d["liver_slab_sd"] = float(np.std(slab)) if len(slab) > 1 else np.nan
    sp = u == names.index("spleen") + 1
    d["spleen_vox"] = int(sp.sum())
    d["spleen_med"] = med(hu, ndimage.binary_erosion(sp, iterations=1))
    d["aorta_med"] = med(hu, u == names.index("aorta") + 1)
    for k in ("kidney_left", "kidney_right"):
        km = u == names.index(k) + 1
        if km.sum() >= 50:
            q = np.percentile(hu[km], [10, 50, 90])
            d[f"{k}_p10"], d[f"{k}_med"], d[f"{k}_p90"] = map(float, q)
    # full TS mask: portal vein / IVC / L3 level
    try:
        m = np.asarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
        if m.shape != hu.shape:
            raise ValueError("shape")
        d["mask_agree"] = float(((m == TS["liver"]) & lv).sum() / max(1, lv.sum()))
        d["portal_med"] = med(hu, ndimage.binary_erosion(m == TS["portal"]), 10)
        d["ivc_med"] = med(hu, ndimage.binary_erosion(m == TS["ivc"]), 10)
        for lvl in ("L3", "L1"):
            vz = np.where((m == TS[lvl]).any((1, 2)))[0]
            if len(vz):
                s = int(np.round(vz.mean()))
                sl = hu[s]
                body = ndimage.binary_fill_holes(ndimage.binary_opening(sl > -500, iterations=2))
                lab, n = ndimage.label(body)
                if n:
                    body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
                fat = body & (sl >= -190) & (sl <= -30)
                # subcutaneous ~ fat within 1 cm..band? use outer ring: body minus body eroded 10 voxels (30mm)
                inner = ndimage.binary_erosion(body, iterations=10)
                d[f"{lvl}_body_cm2"] = float(body.sum() * 0.09)
                d[f"{lvl}_fat_cm2"] = float(fat.sum() * 0.09)
                d[f"{lvl}_fat_outer_cm2"] = float((fat & ~inner).sum() * 0.09)
                d[f"{lvl}_fat_inner_cm2"] = float((fat & inner).sum() * 0.09)
    except Exception as e:
        d["mask_err"] = str(e)[:40]
    return d


if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L))
