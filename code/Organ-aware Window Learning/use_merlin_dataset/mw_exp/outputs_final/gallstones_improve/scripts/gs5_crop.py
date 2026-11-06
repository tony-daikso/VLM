"""GS5 stage A (CPU): gallbladder crop per case from the existing 3 mm prep (hu + units, z,y,x) and the TS --fast 'total' mask
(for the GB-fossa ROI, same definition as analysis_absent_gb/scripts/ag_feat.py: 30 mm ball around the expected GB point
= liver bbox fraction [0.3355, 0.7002, 0.6623] (median of the ag pilot), minus bone(+2)/vessels/uro/bowel(+1)/liver core).
Crop 64^3 voxels (192 mm cube, pad -1024), centred on the GB centroid (GB >= 100 vox) else on the expected fossa point.
Regions (bool, overlapping) used for mean+max pooling by gs5_extract.py (= biliary bd5_extract.py format):
  r0 gb_dil6 : GB mask (prep units) dilated by 2 voxels (Euclidean <= 2 vox = 6 mm)
  r1 gb_dil6 U fossa30
  r2 gb_raw  : the original GB mask (mean+max of the undilated unit)
usage: gs5_crop.py ids.txt out_dir prep_dir[,prep_dir2,...] mask_dir[,mask_dir2,...] [nproc]"""
import sys, os, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
S = 64
EXP = np.array([0.3355, 0.7002, 0.6623])
try:
    from totalsegmentator.map_to_binary import class_map
    CM = class_map["total"]
except Exception:  # fixed TS v2 'total' ids (fallback)
    CM = None
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
BOWEL = ids_of("stomach", "duodenum", "small_bowel", "colon", "esophagus")
URO = ids_of("kidney", "urinary_bladder")
INV = {v: k for k, v in CM.items()}; LIV = INV["liver"]


def crop(a, start, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = a[src]
    return out


def find(sid, dirs, ext):
    return next((f"{d}/{sid}{ext}" for d in dirs if os.path.exists(f"{d}/{sid}{ext}")), None)


def one(sid):
    fo = f"{ODIR}/{sid}.npz"
    if os.path.exists(fo):
        return "skip"
    fp, fm = find(sid, PREP, ".npz"), find(sid, MASK, ".nii.gz")
    if fp is None or fm is None:
        return "nofile"
    z = np.load(fp); hu = z["hu"]; names = list(z["unit_names"])
    gb = z["units"] == names.index("gallbladder") + 1
    m = np.asarray(nib.load(fm).dataobj).astype(np.uint8).transpose(2, 1, 0)
    if m.shape != hu.shape:
        return "shape"
    L = m == LIV; gbv = int(gb.sum())
    pt = None
    if L.sum() >= 1000:
        lz, ly, lx = [np.where(L.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
        lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float)
        pt = lo + EXP * (hi - lo)
    if gbv >= 100:
        cen = np.array(ndimage.center_of_mass(gb))
    elif pt is not None:
        cen = pt
    elif gbv > 0:
        cen = np.array(ndimage.center_of_mass(gb))
    else:
        np.savez_compressed(fo, present=False, gb_vox=gbv); return "none"
    start = [int(round(c)) - S // 2 for c in cen]
    h = crop(hu, start, -1024); mc = crop(m, start, 0); g = crop(gb.astype(np.uint8), start, 0) > 0
    r2 = g
    r0 = (ndimage.distance_transform_edt(~g) <= 2.0) if g.any() else np.zeros_like(g)
    fossa = np.zeros_like(g)
    if pt is not None:
        Lc = mc == LIV
        excl = ndimage.binary_dilation(np.isin(mc, BONE), iterations=2) | np.isin(mc, VESS + URO) | ndimage.binary_dilation(np.isin(mc, BOWEL), iterations=1)
        p = pt - np.array(start)
        zz, yy, xx = np.ogrid[:S, :S, :S]
        dist = np.sqrt((zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2) * 3.0
        fossa = (dist <= 30) & ~excl & ~ndimage.binary_erosion(Lc, iterations=2)
    r1 = r0 | fossa
    np.savez_compressed(fo, present=True, hu=h.astype(np.int16), regions=np.stack([r0, r1, r2]), start=np.array(start),
                        center=np.array(cen), gb_vox=gbv, fossa_vox=int(fossa.sum()))
    return "ok"


def safe(sid):
    try:
        return sid, one(sid)
    except Exception as e:  # noqa: BLE001
        return sid, "err " + repr(e)[:120]


if __name__ == "__main__":
    idf, ODIR = sys.argv[1], sys.argv[2]
    PREP, MASK = sys.argv[3].split(","), sys.argv[4].split(",")
    npr = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    os.makedirs(ODIR, exist_ok=True)
    from collections import Counter
    cnt = Counter()
    L_ = [l.strip() for l in open(idf) if l.strip()]
    with Pool(npr) as p:
        for i, (s, r) in enumerate(p.imap_unordered(safe, L_, chunksize=4)):
            cnt[r.split()[0]] += 1
            if r.startswith("err"): print(s, r, flush=True)
            if (i + 1) % 500 == 0: print(i + 1, len(L_), dict(cnt), flush=True)
    print("CROP_DONE", dict(cnt), flush=True)
