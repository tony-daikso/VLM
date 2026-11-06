"""BD5 stage A (CPU): porta-hepatis crop per case from the existing 3 mm prep (hu, nnU-Net z,y,x order) and the full
TotalSegmentator 'total' mask. Centre = centroid of portal-vein voxels at the liver hilum (PV within 2 voxels of the
liver); fallback = PV voxel closest to the liver centroid. Crop 64^3 voxels (192 mm cube), pad -1024.
Region masks inside the crop (bool, may overlap), used for mask pooling in stage B:
  r0 porta_ball : all voxels within 30 mm of the centre
  r1 hdl        : unlabelled voxels <=24 mm from PV and <=30 mm from liver or pancreas (hepatoduodenal ligament / CBD)
  r2 hilum_liv  : liver voxels <=30 mm from PV (central intrahepatic ducts)
  r3 phead      : pancreas / unlabelled voxels <=9 mm from pancreas and <=15 mm from duodenum (distal CBD)
Usage: bd5_crop.py ids.txt prep_dir mask_dir out_dir   (ids = train/val only). Output out_dir/{id}.npz."""
import sys, os, numpy as np, nibabel as nib
from scipy import ndimage
C = dict(liver=5, panc=7, duod=19, pv=64)
S = 64


def crop(a, start, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = a[src]
    return out


def one(sid, prep, mdir, odir):
    fo = f"{odir}/{sid}.npz"
    if os.path.exists(fo):
        return "skip"
    fp, fm = f"{prep}/{sid}.npz", f"{mdir}/{sid}.nii.gz"
    if not (os.path.exists(fp) and os.path.exists(fm)):
        return "nofile"
    hu = np.load(fp)["hu"]
    m = np.asarray(nib.load(fm).dataobj).astype(np.uint8).T
    if m.shape != hu.shape:
        return "shape"
    L, P, D, PV = (m == C["liver"]), (m == C["panc"]), (m == C["duod"]), (m == C["pv"])
    if PV.sum() < 20 or L.sum() < 500:
        np.savez_compressed(fo, present=False)
        return "nopv"
    dL = ndimage.distance_transform_edt(~L)
    hil = PV & (dL <= 2)
    if hil.sum() >= 5:
        cen = np.argwhere(hil).mean(0)
    else:
        lc = np.argwhere(L).mean(0); pv = np.argwhere(PV)
        cen = pv[np.argmin(((pv - lc) ** 2).sum(1))].astype(float)
    start = [int(round(c)) - S // 2 for c in cen]
    h = crop(hu, start, -1024); mc = crop(m, start, 0)
    Lc, Pc, Dc, PVc = (mc == C["liver"]), (mc == C["panc"]), (mc == C["duod"]), (mc == C["pv"])
    zz, yy, xx = np.indices((S, S, S)); c0 = S // 2
    r0 = ((zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2) <= 10 ** 2
    dpv = ndimage.distance_transform_edt(~PVc) if PVc.any() else np.full(h.shape, 99.)
    dl = ndimage.distance_transform_edt(~Lc) if Lc.any() else np.full(h.shape, 99.)
    dp = ndimage.distance_transform_edt(~Pc) if Pc.any() else np.full(h.shape, 99.)
    dd = ndimage.distance_transform_edt(~Dc) if Dc.any() else np.full(h.shape, 99.)
    free = mc == 0
    r1 = free & (dpv <= 8) & ((dl <= 10) | (dp <= 10))
    r2 = Lc & (dpv <= 10)
    r3 = (Pc | free) & (dp <= 3) & (dd <= 5)
    np.savez_compressed(fo, present=True, hu=h.astype(np.int16), regions=np.stack([r0, r1, r2, r3]),
                        start=np.array(start), center=np.array(cen))
    return "ok"


if __name__ == "__main__":
    ids, prep, mdir, odir = sys.argv[1:5]
    os.makedirs(odir, exist_ok=True)
    from collections import Counter
    cnt = Counter()
    L_ = [l.strip() for l in open(ids) if l.strip()]
    for i, s in enumerate(L_):
        try:
            cnt[one(s, prep, mdir, odir)] += 1
        except Exception as e:  # noqa: BLE001
            cnt["err"] += 1; print(s, repr(e)[:100], flush=True)
        if (i + 1) % 500 == 0:
            print(i + 1, len(L_), dict(cnt), flush=True)
    print("CROP_DONE", dict(cnt), flush=True)
