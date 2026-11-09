"""Kidney / collecting-system HU features from prep npz (3 mm HU + unit map) and the full TS 'total' label map (masks/*.nii.gz).
Usage: hn_hu.py ids.txt out_root out.csv nproc     (out_root has prep/ and masks/)
Per side (l/r): kidney mask stats, fluid (-10..25 HU) voxels inside the kidney mask, in the 'sinus' (closing(kidney) minus kidney),
and in a 9 mm unlabeled perirenal/peri-hilar shell; largest connected fluid component touching the kidney, its AP / transverse extent;
dense voxels (>300 / >1000 HU: excreted contrast, stone, stent); TS kidney_cyst volume; kidney height relative to L3.
Global: bladder volume / HU / dense voxels (excretory phase), aorta HU (phase)."""
import sys, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage
ids, root, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
KID = {"l": 3, "r": 2}; CYST = {"l": 23, "r": 24}; BLADDER = 21; L3 = 29; L5 = 27
FL0, FL1 = -10, 25
BALL = ndimage.generate_binary_structure(3, 1)

def side_feats(hu, lab, s, d):
    K = lab == KID[s]; C = lab == CYST[s]
    d[f"{s}_vox"] = int(K.sum()); d[f"{s}_cyst_vox"] = int(C.sum())
    if K.sum() < 50:
        return
    zz, yy, xx = np.where(K)
    d[f"{s}_z_mm"] = float(zz.mean() * 3.0)
    m = 6
    sl = tuple(slice(max(0, a.min() - m), b.max() + m + 1) for a, b in ((zz, zz), (yy, yy), (xx, xx)))
    h = hu[sl].astype(np.float32); L = lab[sl]; k = K[sl]; c = C[sl]
    v = h[k]
    for p, x in zip((10, 25, 50, 75, 90), np.percentile(v, (10, 25, 50, 75, 90))):
        d[f"{s}_p{p}"] = float(x)
    d[f"{s}_max"] = float(v.max()); d[f"{s}_std"] = float(v.std())
    fl = (h >= FL0) & (h <= FL1)
    d[f"{s}_fl_in"] = int((fl & k).sum()); d[f"{s}_fl_in_frac"] = float((fl & k).mean() / max(k.mean(), 1e-9))
    d[f"{s}_n300_in"] = int((k & (h > 300)).sum()); d[f"{s}_n1000_in"] = int((k & (h > 1000)).sum())
    # sinus: 3D closing (3 vox = 9 mm) + per-slice hole fill, minus kidney; keep unlabeled / cyst voxels
    kc = ndimage.binary_closing(k, BALL, iterations=3)
    kc = np.stack([ndimage.binary_fill_holes(a) for a in kc]) | k
    free = (L == 0) | c
    sinus = kc & ~k & free
    d[f"{s}_sinus_vox"] = int(sinus.sum()); d[f"{s}_sinus_fl"] = int((sinus & fl).sum())
    d[f"{s}_sinus_n300"] = int((sinus & (h > 300)).sum())
    shell = ndimage.binary_dilation(k, BALL, iterations=3) & ~kc & free
    d[f"{s}_shell_fl"] = int((shell & fl).sum()); d[f"{s}_shell_n300"] = int((shell & (h > 300)).sum())
    d[f"{s}_shell_n1000"] = int((shell & (h > 1000)).sum())
    # largest fluid CC within kidney U sinus U shell that touches the kidney (or its sinus)
    F = fl & (k | sinus | shell)
    cl, n = ndimage.label(F)
    if n:
        touch = np.unique(cl[(k | sinus) & F]); touch = touch[touch > 0]
        if len(touch):
            sz = np.bincount(cl.ravel())[touch]; best = touch[sz.argmax()]
            comp = cl == best
            d[f"{s}_flcc"] = int(comp.sum())
            d[f"{s}_flcc_out"] = int((comp & ~k).sum())          # part outside the TS kidney mask
            ap = tr = 0
            for z in np.where(comp.any((1, 2)))[0]:
                a = comp[z]; ys = np.where(a.any(1))[0]; xs = np.where(a.any(0))[0]
                ap = max(ap, ys.max() - ys.min() + 1); tr = max(tr, xs.max() - xs.min() + 1)
            d[f"{s}_flcc_ap_mm"] = float(ap * 3); d[f"{s}_flcc_tr_mm"] = float(tr * 3)
            d[f"{s}_flcc_z_mm"] = float(comp.any((1, 2)).sum() * 3)
    d.setdefault(f"{s}_flcc", 0)

def feat(sid):
    try:
        z = np.load(f"{root}/prep/{sid}.npz")
        names = list(z["unit_names"]); hu = z["hu"]; u = z["units"]
        lab = np.asarray(nib.load(f"{root}/masks/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
    except Exception:
        return None
    if lab.shape != hu.shape:
        return None
    d = {"study_id": sid}
    for s in ("l", "r"):
        side_feats(hu, lab, s, d)
    am = u == names.index("aorta") + 1
    d["aorta_hu_med"] = float(np.median(hu[am])) if am.sum() > 50 else np.nan
    B = lab == BLADDER
    d["bl_vox"] = int(B.sum())
    if B.sum() > 50:
        bv = hu[B].astype(np.float32)
        d["bl_p50"] = float(np.median(bv)); d["bl_p90"] = float(np.percentile(bv, 90)); d["bl_n200"] = int((bv > 200).sum())
    for nm, li in (("L3", L3), ("L5", L5)):
        mm = lab == li
        d[f"{nm}_z_mm"] = float(np.where(mm)[0].mean() * 3.0) if mm.sum() > 20 else np.nan
    return d

if __name__ == "__main__":
    Lst = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, Lst, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(Lst))
