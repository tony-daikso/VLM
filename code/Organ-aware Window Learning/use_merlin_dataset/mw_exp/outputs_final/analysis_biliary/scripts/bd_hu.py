"""Handcrafted biliary-dilation proxies from prep npz (3mm HU, z,y,x) + full TotalSegmentator 'total' mask (x,y,z, 3mm).
Usage: bd_hu.py ids.txt prep_dir mask_dir out.csv nproc   (ids = train/val only)
No bile-duct class exists, so all features are indirect: low-attenuation voxels in the liver / hilum,
fluid-attenuation unlabelled voxels around the portal vein (porta hepatis, hepatoduodenal ligament) and the
pancreatic head, air in the liver (pneumobilia), metal near the porta (stent/drain), phase markers."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage
import nibabel as nib
ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
C = dict(gb=4, liver=5, stomach=6, panc=7, duod=19, aorta=52, ivc=63, pv=64, spleen=1)
ball = lambda r: (np.add.outer(np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2), np.arange(-r, r + 1) ** 2) <= r * r)


def med(hu, m):
    return float(np.median(hu[m])) if m.sum() > 20 else np.nan


def ccstats(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return 0, 0
    s = np.bincount(lab.ravel())[1:]
    return int(s.max()), int((s >= 3).sum())


def feat(sid):
    fp, fm = f"{prep}/{sid}.npz", f"{mdir}/{sid}.nii.gz"
    if not (os.path.exists(fp) and os.path.exists(fm)):
        return None
    try:
        hu = np.load(fp)["hu"].astype(np.float32)
        m = np.asarray(nib.load(fm).dataobj).astype(np.uint8).T
    except Exception:
        return None
    if m.shape != hu.shape:
        return None
    d = {"study_id": sid}
    M = {k: m == v for k, v in C.items()}
    for k in ("liver", "panc", "gb", "duod", "pv"):
        d[f"{k}_vox"] = int(M[k].sum())
    for k in ("aorta", "pv", "ivc", "liver", "spleen", "panc"):
        d[f"{k}_hu"] = med(hu, M[k])
    L = M["liver"]
    if L.sum() < 500:
        return d
    lmed = d["liver_hu"]
    Le = ndimage.binary_erosion(L, iterations=1)
    d["liver_hu_sd"] = float(hu[Le].std()) if Le.sum() > 20 else np.nan
    # relative low attenuation (phase-robust): >=35 HU below liver median, above fat
    low = Le & (hu < lmed - 35) & (hu > -20)
    d["liv_low_n"] = int(low.sum()); d["liv_low_frac"] = float(low.sum() / max(Le.sum(), 1))
    d["liv_low_maxcc"], d["liv_low_ncc"] = ccstats(low)
    fl = Le & (hu >= -5) & (hu <= 25)
    d["liv_fluid_n"] = int(fl.sum())
    d["liv_air_n"] = int((L & (hu < -300)).sum())                     # pneumobilia proxy
    # hilum: liver voxels within ~21 mm of the portal vein mask (or the liver's medial-inferior third if no PV)
    pv = M["pv"]
    if pv.sum() >= 20:
        pvd = ndimage.binary_dilation(pv, structure=ball(7))
        hil = Le & pvd & ~pv
        d["hil_vox"] = int(hil.sum())
        hlow = hil & (hu < lmed - 35) & (hu > -20)
        d["hil_low_n"] = int(hlow.sum()); d["hil_low_frac"] = float(hlow.sum() / max(hil.sum(), 1))
        d["hil_low_maxcc"], _ = ccstats(hlow)
        # extrahepatic porta / hepatoduodenal ligament: unlabelled voxels within 15 mm of PV and 30 mm of liver or pancreas
        near = ndimage.binary_dilation(pv, structure=ball(5)) & (m == 0)
        near &= ndimage.binary_dilation(L | M["panc"], structure=ball(10))
        d["porta_vox"] = int(near.sum())
        pf = near & (hu >= -5) & (hu <= 30)
        d["porta_fluid_n"] = int(pf.sum()); d["porta_fluid_frac"] = float(pf.sum() / max(near.sum(), 1))
        d["porta_fluid_maxcc"], _ = ccstats(pf)
        d["porta_air_n"] = int((near & (hu < -300)).sum())
        d["porta_metal_n"] = int((ndimage.binary_dilation(pv, structure=ball(8)) & (m == 0) & (hu > 600)).sum())
        d["porta_hu_med"] = med(hu, near)
    # pancreatic head region: unlabelled voxels within 9 mm of both pancreas and duodenum (distal CBD territory)
    P, D = M["panc"], M["duod"]
    if P.sum() >= 50:
        Pe = ndimage.binary_erosion(P, iterations=1)
        pmed = d["panc_hu"]
        d["panc_low_n"] = int((Pe & (hu < pmed - 30) & (hu > -20)).sum())    # dilated PD / intrapancreatic CBD / mass
        d["panc_low_frac"] = float(d["panc_low_n"] / max(Pe.sum(), 1))
        if D.sum() >= 50:
            reg = ndimage.binary_dilation(P, structure=ball(3)) & ndimage.binary_dilation(D, structure=ball(3)) & (m == 0)
            d["phead_vox"] = int(reg.sum())
            d["phead_fluid_n"] = int((reg & (hu >= -5) & (hu <= 30)).sum())
            d["phead_metal_n"] = int((reg & (hu > 600)).sum())
    return d


if __name__ == "__main__":
    L_ = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L_, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L_), flush=True)
