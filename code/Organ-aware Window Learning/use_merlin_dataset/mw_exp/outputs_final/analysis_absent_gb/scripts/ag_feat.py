"""GB / GB-fossa / surgical-clip features from the TS --fast multilabel mask (3 mm) + prep HU.
usage: ag_feat.py ids.txt prep_dir mask_dir out.csv nproc [pilot]
pilot: writes GB centroid as fraction of the liver bounding box (to define the expected GB-fossa point)."""
import sys, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage
from totalsegmentator.map_to_binary import class_map
CM = class_map["total"]; INV = {v: k for k, v in CM.items()}
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
BOWEL = ids_of("stomach", "duodenum", "small_bowel", "colon", "esophagus")
URO = ids_of("kidney", "urinary_bladder")
GB, LIV = INV["gallbladder"], INV["liver"]
# expected GB centroid as fraction of liver bbox (z,y,x), median of train GB-present pilot cases (filled after pilot)
EXP = [float(x) for x in (sys.argv[7].split(",") if len(sys.argv) > 7 else ["nan"] * 3)]
pilot = len(sys.argv) > 6 and sys.argv[6] == "pilot"
def feat(sid):
    try:
        hu = np.load(f"{prep}/{sid}.npz")["hu"].astype(np.float32)
        m = np.asarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
    except Exception as e:
        return None
    if m.shape != hu.shape: return None
    d = {"study_id": sid}
    g = m == GB; L = m == LIV
    d["gb_vox"] = int(g.sum()); d["liver_vox"] = int(L.sum())
    if L.sum() < 1000: return d
    lz, ly, lx = [np.where(L.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float)
    if g.sum() > 0:
        lab, n = ndimage.label(g); sz = np.bincount(lab.ravel())[1:]
        d["gb_ncc"] = int(n); d["gb_lcc_frac"] = float(sz.max() / sz.sum())
        c = np.array(ndimage.center_of_mass(g)); d.update({f"gb_frac_{a}": float(v) for a, v in zip("zyx", (c - lo) / np.maximum(hi - lo, 1))})
        v = hu[g]; d["gb_hu_med"] = float(np.median(v)); d["gb_hu_mean"] = float(v.mean()); d["gb_hu_std"] = float(v.std())
        d["gb_f_fluid"] = float(((v > -10) & (v < 40)).mean()); d["gb_f_fat"] = float(((v > -200) & (v < -30)).mean())
        d["gb_f_air"] = float((v < -500).mean()); d["gb_f_dense"] = float((v > 100).mean())
        # GB-liver contact: fraction of GB surface touching liver (a leak onto bowel/fat has low contact)
        sh = ndimage.binary_dilation(g) & ~g
        d["gb_liver_contact"] = float((L & sh).sum() / max(sh.sum(), 1))
        d["gb_bowel_contact"] = float(np.isin(m[sh], BOWEL).mean())
    if pilot: return d
    excl = ndimage.binary_dilation(np.isin(m, BONE), iterations=2) | np.isin(m, VESS + URO) | ndimage.binary_dilation(np.isin(m, BOWEL), iterations=1)
    pt = lo + np.array(EXP) * (hi - lo)
    zz, yy, xx = np.ogrid[:m.shape[0], :m.shape[1], :m.shape[2]]
    dist = np.sqrt((zz - pt[0]) ** 2 + (yy - pt[1]) ** 2 + (xx - pt[2]) ** 2) * 3.0  # mm
    for r in (30, 50):
        roi = (dist <= r) & ~excl & ~ndimage.binary_erosion(L, iterations=2)  # keep liver rim (clips lie on the liver surface)
        v = hu[roi]; d[f"f{r}_n"] = int(roi.sum())
        for t in (300, 500, 800, 1200):
            d[f"f{r}_n{t}"] = int((v > t).sum())
        d[f"f{r}_max"] = float(v.max()) if v.size else np.nan
        if r == 30:
            lab, n = ndimage.label(roi & (hu > 500)); d["f30_blobs500"] = int(n)
            nl = roi & ~L & ~g
            w = hu[nl]; d["f30_nonliver"] = int(nl.sum())
            d["f30_f_fat"] = float(((w > -200) & (w < -30)).mean()) if w.size else np.nan
            d["f30_f_fluid"] = float(((w > -10) & (w < 40)).mean()) if w.size else np.nan
            d["f30_f_soft"] = float(((w >= 40) & (w < 120)).mean()) if w.size else np.nan
            d["f30_gb_frac"] = float(g[dist <= r].mean())
    # whole-scan non-bone metal (any surgery / device proxy)
    nb = ~ndimage.binary_dilation(np.isin(m, BONE), iterations=2)
    d["body_metal1500"] = int(((hu > 1500) & nb).sum())
    # perihepatic shell (<=9 mm outside liver), clip-level density
    shell = ndimage.binary_dilation(L, iterations=3) & ~L & ~excl
    d["peri_n800"] = int((hu[shell] > 800).sum()); d["peri_n1200"] = int((hu[shell] > 1200).sum())
    return d
if __name__ == "__main__":
    idf, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    L = [l.strip() for l in open(idf) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), flush=True)
