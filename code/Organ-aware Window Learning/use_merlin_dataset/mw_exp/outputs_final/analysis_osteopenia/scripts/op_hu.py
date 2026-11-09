"""Osteopenia bone-HU / phase features from prep npz (3 mm HU, z-y-x) + TS total masks (3 mm, RAS x-y-z -> transpose(2,1,0)).
Usage: op_hu.py ids.txt prep_dir mask_dir out.csv nproc   (ids must already be train/val only)
Per level T12..L5: whole-vertebra median/mean HU, vertebral body (anterior 40% of AP extent, as fracture_features.py),
trabecular core = body eroded 1 voxel in-plane (removes ~3 mm cortex), cement voxels (>1000 HU) in body.
References: aorta / IVC / portal median HU (phase), autochthon + iliopsoas median HU (muscle)."""
import os, sys, numpy as np, pandas as pd, nibabel as nib
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from scipy import ndimage
ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
LV = {"T12": 32, "L1": 31, "L2": 30, "L3": 29, "L4": 28, "L5": 27}
REF = {"aorta": [52], "ivc": [63], "portal": [64], "autochthon": [86, 87], "psoas": [88, 89]}
ST = np.zeros((1, 3, 3), bool); ST[0, 1, :] = True; ST[0, :, 1] = True


def feat(sid):
    try:
        hu = np.load(f"{prep}/{sid}.npz")["hu"].astype(np.float32)
        lab = np.asanyarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
    except Exception:
        return None
    if lab.shape != hu.shape:
        return {"study_id": sid, "shape_mismatch": 1}
    d = {"study_id": sid, "shape_mismatch": 0}
    for k, cl in REF.items():
        m = np.isin(lab, cl)
        if k in ("aorta", "ivc", "portal") and m.sum() > 30:
            m = ndimage.binary_erosion(m, ST) if ndimage.binary_erosion(m, ST).sum() > 20 else m
        d[f"{k}_hu"] = float(np.median(hu[m])) if m.sum() > 20 else np.nan
    for lv, cl in LV.items():
        m = lab == cl
        n = int(m.sum()); d[f"{lv}_vox"] = n
        if n < 150:
            continue
        v = hu[m]
        d[f"{lv}_whole_med"] = float(np.median(v)); d[f"{lv}_whole_mean"] = float(v.mean())
        idx = np.argwhere(m); y0, y1 = idx[:, 1].min(), idx[:, 1].max()
        b = idx[idx[:, 1] >= y1 - 0.4 * (y1 - y0)]
        bm = np.zeros_like(m); bm[b[:, 0], b[:, 1], b[:, 2]] = True
        bv = hu[bm]
        d[f"{lv}_body_med"] = float(np.median(bv)); d[f"{lv}_body_p10"] = float(np.percentile(bv, 10))
        d[f"{lv}_body_n1000"] = int((bv > 1000).sum())
        core = ndimage.binary_erosion(bm, ST)
        d[f"{lv}_core_n"] = int(core.sum())
        if core.sum() >= 10:
            cv = hu[core]
            d[f"{lv}_core_med"] = float(np.median(cv)); d[f"{lv}_core_mean"] = float(cv.mean())
            d[f"{lv}_core_p25"] = float(np.percentile(cv, 25))
        d[f"{lv}_zc"] = float(b[:, 0].mean())
    return d


if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    res = []
    with Pool(npr) as p:
        for j, r in enumerate(p.imap_unordered(feat, L, chunksize=4)):
            if r is not None:
                res.append(r)
            if j % 250 == 0:
                print(f"[{j}/{len(L)}]", flush=True)
                pd.DataFrame(res).to_csv(out + ".partial", index=False)
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), flush=True)
