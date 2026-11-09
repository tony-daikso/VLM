"""Ascites handcrafted features from prep npz (3 mm HU, z-y-x) + TS 'total' masks (3 mm, x-y-z -> transpose(2,1,0)).
No embeddings, no labels. Usage: asc_hu.py ids.txt outputs_dir out.csv nproc   (ids are train/val only)
Peritoneal cavity proxy = per-slice 2D convex hull of abdominal organs/bones (liver top .. pelvis floor), dilated 6 mm,
inside the body, minus every TS label (dilated 3 mm). Fluid = 3x3x3-median-filtered HU in [0,30].
Lung-adjacent voxels (<=9 mm from a lung lobe = pleural space / posterior costophrenic recess) are counted separately."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage
from skimage.morphology import convex_hull_image

LUNG = [10, 11, 12, 13, 14]
HULL = [1, 2, 3, 4, 5, 6, 7, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 52, 63, 64, 65, 66, 67, 68, 77, 78, 88, 89]
def lut(ks):
    t = np.zeros(256, bool); t[ks] = True; return t
LUT_LUNG, LUT_HULL, LUT_PELV, LUT_BOWEL = lut(LUNG), lut(HULL), lut([77, 78, 21]), lut([18, 19, 20])
B2 = ndimage.generate_binary_structure(2, 1)
B3 = ndimage.generate_binary_structure(3, 1)


def shell(m, k, excl):
    return ndimage.binary_dilation(m, B3, iterations=k) & ~m & ~excl


def feat(args):
    sid, out = args
    try:
        hu = np.load(f"{out}/prep/{sid}.npz")["hu"].astype(np.float32)
        lab = np.asanyarray(nib.load(f"{out}/masks/{sid}.nii.gz").dataobj).transpose(2, 1, 0).astype(np.uint8)
    except Exception:
        return None
    if hu.shape != lab.shape:
        return None
    d = {"study_id": sid, "nz": hu.shape[0]}
    V = 0.027  # ml per voxel
    # crop: abdomen z range (liver top .. hips/bladder bottom, +-3 slices) and the body bounding box
    zl = np.where((lab == 5).any((1, 2)))[0]
    zh = np.where(LUT_PELV[lab].any((1, 2)))[0]
    z0 = max(int(zh[0]) - 3, 0) if len(zh) else 0
    z1 = min(int(zl[-1]) + 4, hu.shape[0]) if len(zl) else hu.shape[0]
    d["liver_cut_top"] = int(len(zl) > 0 and zl[-1] == hu.shape[0] - 1)
    hu, lab = hu[z0:z1], lab[z0:z1]
    pr = (hu > -300)
    yy = np.where(pr.any((0, 2)))[0]; xx = np.where(pr.any((0, 1)))[0]
    if len(yy) and len(xx):
        hu = hu[:, max(yy[0] - 2, 0):yy[-1] + 3, max(xx[0] - 2, 0):xx[-1] + 3]
        lab = lab[:, max(yy[0] - 2, 0):yy[-1] + 3, max(xx[0] - 2, 0):xx[-1] + 3]
    hs = ndimage.median_filter(hu.astype(np.int16), size=3).astype(np.float32)
    body = hu > -300
    lb, n = ndimage.label(body)
    if n:
        body = lb == (np.bincount(lb.ravel())[1:].argmax() + 1)
    body = np.stack([ndimage.binary_fill_holes(s) for s in body])
    org = lab > 0
    orgd = ndimage.binary_dilation(org, B3, iterations=1)
    lung = LUT_LUNG[lab]
    lungnear = ndimage.binary_dilation(lung, B3, iterations=3) if lung.any() else np.zeros_like(lung)
    liver = lab == 5; spleen = lab == 1; bladder = lab == 21
    d["liver_vox"] = int(liver.sum()); d["spleen_vox"] = int(spleen.sum()); d["bladder_vox"] = int(bladder.sum())
    for nm, k in (("liver", 5), ("spleen", 1), ("aorta", 52), ("portal", 64), ("kidney_r", 2)):
        mm = lab == k
        d[f"{nm}_hu"] = float(np.median(hu[mm])) if mm.sum() > 30 else np.nan
    zl = np.where(liver.any((1, 2)))[0]
    zh = np.where(LUT_PELV[lab].any((1, 2)))[0]
    z5 = np.where((lab == 27).any((1, 2)))[0]
    ztop = int(zl[-1]) if len(zl) else hu.shape[0] - 1
    zbot = int(zh[0]) if len(zh) else 0
    d["pelvis_cov"] = int(len(zh) > 0 and bladder.sum() > 100)
    d["abd_z_mm"] = 3 * (ztop - zbot + 1)
    hullm = np.zeros_like(body)
    hs_set = LUT_HULL[lab]
    for z in range(zbot, ztop + 1):
        if hs_set[z].sum() >= 10:
            h = convex_hull_image(hs_set[z])
            hullm[z] = ndimage.binary_dilation(h, B2, iterations=2)
    hullm &= body
    cav = hullm & ~orgd
    cav_pl = cav & ~lungnear
    pleural = cav & lungnear
    d["cav_vox"] = int(cav_pl.sum())
    fl = (hs >= 0) & (hs <= 30)
    for lo, hi in ((-20, 0), (0, 15), (15, 30), (30, 45), (45, 70)):
        d[f"cav_ml_{lo}_{hi}"] = float(((hs >= lo) & (hs < hi) & cav_pl).sum() * V)
    f = fl & cav_pl
    d["fluid_ml"] = float(f.sum() * V)
    d["fluid_frac"] = float(f.sum() / max(cav_pl.sum(), 1))
    lf, nf = ndimage.label(f, ndimage.generate_binary_structure(3, 3))
    if nf:
        sz = np.bincount(lf.ravel())[1:]
        d["fluid_maxcc_ml"] = float(sz.max() * V); d["fluid_cc10_ml"] = float(sz[sz >= 10].sum() * V)
        d["fluid_ncc10"] = int((sz >= 10).sum())
        big = np.isin(lf, np.where(sz >= 10)[0] + 1)
        d["fluid_cc10_hu"] = float(hu[big].mean()) if big.any() else np.nan
    else:
        d.update(fluid_maxcc_ml=0.0, fluid_cc10_ml=0.0, fluid_ncc10=0, fluid_cc10_hu=np.nan)
    d["pleural_fluid_ml"] = float((fl & pleural).sum() * V)
    hemo = (hs > 30) & (hs <= 70) & cav_pl
    d["hd_fluid_ml"] = float(hemo.sum() * V)
    # perihepatic / perisplenic rims (3-9 mm from organ, not in any label, not lung-adjacent)
    for nm, m in (("liver", liver), ("spleen", spleen)):
        if m.sum() > 100:
            sh = shell(m, 3, org) & body & ~lungnear
            d[f"peri{nm}_ml"] = float((sh & fl).sum() * V)
            d[f"peri{nm}_frac"] = float((sh & fl).sum() / max(sh.sum(), 1))
            d[f"peri{nm}_hu_med"] = float(np.median(hs[sh])) if sh.any() else np.nan
        else:
            d[f"peri{nm}_ml"] = d[f"peri{nm}_frac"] = d[f"peri{nm}_hu_med"] = np.nan
    # pelvis: below the top of L5 (or of the hips), and the bladder rim (cul-de-sac / rectovesical pouch)
    zp = int(z5[-1]) if len(z5) else (int(zh[-1]) if len(zh) else -1)
    if zp >= 0 and len(zh):
        pm = np.zeros_like(cav_pl); pm[:zp + 1] = True
        d["pelvic_fluid_ml"] = float((f & pm).sum() * V)
        d["upper_fluid_ml"] = float((f & ~pm).sum() * V)
    else:
        d["pelvic_fluid_ml"] = np.nan; d["upper_fluid_ml"] = np.nan
    if bladder.sum() > 100:
        sh = shell(bladder, 3, org) & body
        d["peribladder_ml"] = float((sh & fl).sum() * V); d["peribladder_frac"] = float((sh & fl).sum() / max(sh.sum(), 1))
    else:
        d["peribladder_ml"] = d["peribladder_frac"] = np.nan
    # subcutaneous / outside the cavity (anasarca): body minus hull, abdomen z range, no labels
    sub = body & ~ndimage.binary_dilation(hullm, B3, iterations=1) & ~orgd
    sub[:zbot] = False; sub[ztop + 1:] = False
    v = hs[sub]
    fat = v[(v > -250) & (v < -30)]
    d["subq_fat_hu"] = float(np.median(fat)) if fat.size > 100 else np.nan
    d["subq_edema_frac"] = float(((v > -60) & (v < 20)).sum() / max(v.size, 1))
    d["subq_fat_ml"] = float(fat.size * V)
    cv = hs[cav_pl]
    mf = cv[(cv > -250) & (cv < -30)]
    d["mes_fat_hu"] = float(np.median(mf)) if mf.size > 100 else np.nan
    d["mes_fat_ml"] = float(mf.size * V)
    # bowel content: fluid-density voxels inside the bowel masks (unopacified fluid-filled bowel = mimic)
    bw = LUT_BOWEL[lab]
    d["bowel_ml"] = float(bw.sum() * V)
    d["bowel_fluid_ml"] = float((bw & fl).sum() * V)
    d["bowel_dense_ml"] = float((bw & (hs > 150)).sum() * V)   # positive oral contrast
    d["bowel_gas_ml"] = float((bw & (hs < -500)).sum() * V)
    return d


if __name__ == "__main__":
    ids, out, csv, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    L = [l.strip() for l in open(ids) if l.strip()]
    res = []
    if os.path.exists(csv):   # resume: keep finished rows, skip their ids
        old = pd.read_csv(csv); res = old.to_dict("records"); done = set(old.study_id); L = [s for s in L if s not in done]
        print("resume: done", len(done), "todo", len(L), flush=True)
    with Pool(npr) as p:
        for i, r in enumerate(p.imap_unordered(feat, [(s, out) for s in L], chunksize=4)):
            if r is not None:
                res.append(r)
            if (i + 1) % 250 == 0:
                pd.DataFrame(res).to_csv(csv, index=False); print("progress", i + 1, flush=True)
    pd.DataFrame(res).to_csv(csv, index=False)
    print("done", len(res), "of", len(L), flush=True)
