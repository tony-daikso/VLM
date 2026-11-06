"""Exp N region crop units, stage A (CPU). Same pipeline / output format as Exp M ru_crop.py (read by ru_qc.py and
nu_extract.py). Input: 3 mm prep npz (hu, units, affine, unit_names; RAS, array z,y,x: +z superior, +y anterior) +
TS 'total' 3 mm label map (nii, transposed to z,y,x).
Units (--unit):
  gs  : gallbladder / GB-fossa unit, one 64^3 crop (192 mm). Centre = GB centroid (prep 'gallbladder' unit, >= 100 vox),
        else the expected GB-fossa point used by GS5/ag_feat (liver bbox fraction [0.3355, 0.7002, 0.6623]), else the
        GB centroid of a tiny GB. GS6: if GS6_SEG=<dir> holds <id>.nii.gz (1.5 mm full-res GB re-segmentation of the
        GB < 100 vox cases), the GB is replaced by that mask resampled to the 3 mm grid (same rule as gs6_fix.py:
        linear, fraction >= 0.5) before centring and regions (method 'gs6_*').
        regions: r0 GB dilated 2 vox (EDT <= 2 vox = 6 mm); r1 GB-fossa ROI (30 mm ball around the expected fossa point
                 minus bone(+2)/vessels/uro/bowel(+1)/liver core, = GS5 fossa30); r2 dependent (posterior) third of the
                 GB (GB voxels with y <= 33rd percentile of the GB's y; supine -> posterior = dependent).
  ath : aorto-iliac wall unit, K=4 crops of 48^3 (144 mm): c0 'infrarenal' aorta (aorta below the mean kidney-centroid
        level, excluding the lowest 39 mm), c1 'bifurcation' (lowest 39 mm of the aorta), c2 'cia_right' / c3 'cia_left'
        (iliac artery voxels from 21 mm above to 60 mm below the aortic bifurcation level = common iliac segment).
        regions: r0 vessel-wall shell = dilate(aorta|iliac arteries, 2 vox) minus erode(lumen mask, 1 vox), minus bone
                 dilated 1 vox; r1 whole crop.
  fst : focal-steatosis liver regions, K=3 crops of 48^3 (144 mm) at the typical focal fat / sparing sites:
        c0 'falciform' = segment IV anterior: liver voxels within +-30 mm (LR) of the spine midline and in the anterior half
           of the liver (FL feature definition of hs_focal_hu.py) -> centroid;
        c1 'gbfossa' = liver voxels within 15 mm of the GB (prep unit; GB < 100 vox -> 30 mm of the expected fossa point);
        c2 'porta' = liver voxels within 15 mm of the portal vein (TS portal_vein_and_splenic_vein inside/next to the liver).
        regions: r0 site liver = the site's liver voxels (eroded 1 vox) ; r1 local liver = liver (eroded 1 vox) within
                 30 mm of the crop centre; r2 all liver in the crop (eroded 1 vox).
Per crop scalar HU features hu_feats [K,R,5]: n >= 130 HU, n >= 300 HU, max, p99, median.
Output out_dir/{id}.npz: present [K], hu int16 [K,S,S,S], regions packbits of bool [K,R,S,S,S], rshape, center [K,3],
method [K], hu_feats, orient (lr=2, sr=1, ap=1, sa=1, ss=1: RAS prep).
Usage: nu_crop.py UNIT ids.txt out_dir nproc prep1:mask1 [prep2:mask2 ...]   (ids = train/val only)"""
import sys, os, json, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
from totalsegmentator.map_to_binary import class_map

CM = class_map["total"]; INV = {v: k for k, v in CM.items()}
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE_IDS = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
BOWEL = ids_of("stomach", "duodenum", "small_bowel", "colon", "esophagus")
URO = ids_of("kidney", "urinary_bladder")
LIV, AO, ILL, ILR, KR, KL = INV["liver"], INV["aorta"], INV["iliac_artery_left"], INV["iliac_artery_right"], INV["kidney_right"], INV["kidney_left"]
BONE = np.zeros(256, bool); BONE[BONE_IDS] = True
EXP = np.array([0.3355, 0.7002, 0.6623])
BALL1 = ndimage.generate_binary_structure(3, 1)
CFG = {"gs": dict(S=64, K=1, R=3, names=["gb"]),
       "ath": dict(S=48, K=4, R=2, names=["infrarenal", "bifurcation", "cia_right", "cia_left"]),
       "fst": dict(S=48, K=3, R=3, names=["falciform", "gbfossa", "porta"])}
PORTAL = INV["portal_vein_and_splenic_vein"]
SPINE_IDS = ids_of("vertebrae_L", "vertebrae_T1")
GS6 = os.environ.get("GS6_SEG", "")
ORIENT = np.array([2, 1, 1, 1, 1])


def crop(a, start, S, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    if all(d.stop > d.start for d in dst):
        out[dst] = a[src]
    return out


def hu_feats(h, R):
    K, Rn = R.shape[:2]; hf = np.full((K, Rn, 5), np.nan, np.float32)
    for k in range(K):
        for r in range(Rn):
            v = h[k][R[k, r]].astype(np.float32)
            if v.size < 5:
                continue
            hf[k, r] = [(v >= 130).sum(), (v >= 300).sum(), v.max(), np.percentile(v, 99), np.median(v)]
    return hf


# ------------------------------------------------------------------ gallbladder / GB fossa
def gs6_gb(sid, z):
    f = f"{GS6}/{sid}.nii.gz"
    if not GS6 or not os.path.exists(f):
        return None
    from nibabel.processing import resample_from_to
    seg = nib.load(f); g15 = (np.asanyarray(seg.dataobj) == 1)
    shp_xyz = z["units"].shape[::-1]
    r = resample_from_to(nib.Nifti1Image(g15.astype(np.float32), seg.affine), (shp_xyz, z["affine"]), order=1, cval=0.0)
    return (np.asanyarray(r.dataobj) >= 0.5).transpose(2, 1, 0)


def unit_gs(sid, z, hu, m, S):
    names = list(z["unit_names"]); gb = z["units"] == names.index("gallbladder") + 1
    tag = ""
    g6 = gs6_gb(sid, z)
    if g6 is not None:
        gb = g6; tag = "gs6_"
    L = m == LIV; gbv = int(gb.sum()); pt = None
    if L.sum() >= 1000:
        lz, ly, lx = [np.where(L.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
        lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float)
        pt = lo + EXP * (hi - lo)
    if gbv >= 100:
        cen, meth = np.array(ndimage.center_of_mass(gb)), "gb"
    elif pt is not None:
        cen, meth = pt, "fossa"
    elif gbv > 0:
        cen, meth = np.array(ndimage.center_of_mass(gb)), "gbtiny"
    else:
        return [None]
    start = [int(round(c)) - S // 2 for c in cen]
    h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0); g = crop(gb.astype(np.uint8), start, S, 0) > 0
    r0 = (ndimage.distance_transform_edt(~g) <= 2.0) if g.any() else np.zeros_like(g)
    fossa = np.zeros_like(g)
    if pt is not None:
        Lc = mc == LIV
        excl = ndimage.binary_dilation(BONE[mc], iterations=2) | np.isin(mc, VESS + URO) | ndimage.binary_dilation(np.isin(mc, BOWEL), iterations=1)
        p = pt - np.array(start)
        zz, yy, xx = np.ogrid[:S, :S, :S]
        dist = np.sqrt((zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2) * 3.0
        fossa = (dist <= 30) & ~excl & ~ndimage.binary_erosion(Lc, iterations=2)
    r2 = np.zeros_like(g)
    if g.sum() >= 3:
        yy = np.argwhere(g)[:, 1]; lo_y = np.percentile(yy, 33.3)
        r2 = g & (np.arange(S)[None, :, None] <= lo_y)
    return [(h, np.stack([r0, fossa, r2]), cen, tag + meth)]


# ------------------------------------------------------------------ aorto-iliac wall
def unit_ath(sid, z, hu, m, S):
    A = m == AO
    if A.sum() < 200:
        return [None] * 4
    az = np.argwhere(A)
    zb = az[:, 0].min()                                   # lowest aorta slice = bifurcation (RAS: +z superior)
    kc = [np.argwhere(m == k)[:, 0].mean() for k in (KR, KL) if (m == k).sum() >= 300]
    zk = np.mean(kc) if kc else np.percentile(az[:, 0], 40)
    infra = az[az[:, 0] <= zk]
    if len(infra) < 50:
        infra = az[az[:, 0] <= np.percentile(az[:, 0], 40)]
    cents = []
    up = infra[infra[:, 0] > zb + 13]
    cents.append((up.mean(0), "infrarenal") if len(up) >= 30 else ((infra.mean(0), "infra_short") if len(infra) >= 30 else None))
    bif = az[az[:, 0] <= zb + 13]
    cents.append((bif.mean(0), "bifurcation") if len(bif) >= 20 else None)
    for lab in (ILR, ILL):
        il = np.argwhere(m == lab); il = il[(il[:, 0] >= zb - 20) & (il[:, 0] <= zb + 7)]
        cents.append((il.mean(0), "cia") if len(il) >= 20 else None)
    out = []
    for cc in cents:
        if cc is None:
            out.append(None); continue
        c, meth = cc
        start = [int(round(x)) - S // 2 for x in c]
        h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0)
        V = np.isin(mc, [AO, ILL, ILR])
        shell = ndimage.binary_dilation(V, BALL1, iterations=2) & ~ndimage.binary_erosion(V, BALL1, iterations=1) \
            & ~ndimage.binary_dilation(BONE[mc], BALL1, iterations=1)
        out.append((h, np.stack([shell, np.ones_like(V)]), c, meth))
    return out


# ------------------------------------------------------------------ focal steatosis liver sites
def unit_fst(sid, z, hu, m, S):
    L = m == LIV
    if L.sum() < 1000:
        return [None] * 3
    names = list(z["unit_names"]); gb = z["units"] == names.index("gallbladder") + 1
    Lz = np.argwhere(L)
    lo, hi = Lz.min(0).astype(float), Lz.max(0).astype(float)
    sites = []
    # c0 falciform / segment IV anterior: LR (x, axis 2) within 10 vox of the spine midline, anterior half of the liver (y)
    sp = np.argwhere(np.isin(m, SPINE_IDS))
    mid = np.median(sp[:, 2]) if len(sp) >= 50 else None
    if mid is not None:
        ymid = np.median(Lz[:, 1])
        sel = Lz[(np.abs(Lz[:, 2] - mid) <= 10) & (Lz[:, 1] >= ymid)]
        sites.append((sel, "falciform") if len(sel) >= 30 else None)
    else:
        sites.append(None)
    # c1 GB fossa liver
    sub = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    if gb.sum() >= 100:
        d = ndimage.distance_transform_edt(~gb[sub]); near = (d <= 5) & L[sub]; meth = "gb"
    else:
        pt = lo + EXP * (hi - lo) - lo
        zz, yy, xx = np.ogrid[tuple(slice(0, int(b - a) + 1) for a, b in zip(lo, hi))]
        near = (((zz - pt[0]) ** 2 + (yy - pt[1]) ** 2 + (xx - pt[2]) ** 2) <= 100) & L[sub]; meth = "fossa"
    sel = np.argwhere(near) + lo.astype(int)
    sites.append((sel, meth) if len(sel) >= 30 else None)
    # c2 porta: liver within 15 mm of the portal vein
    pv = m[sub] == PORTAL
    if pv.sum() >= 20:
        d = ndimage.distance_transform_edt(~pv); sel = np.argwhere((d <= 5) & L[sub]) + lo.astype(int)
        sites.append((sel, "porta") if len(sel) >= 30 else None)
    else:
        sites.append(None)
    out = []
    G = np.indices((S, S, S)); c0 = S // 2
    d2 = (G[0] - c0) ** 2 + (G[1] - c0) ** 2 + (G[2] - c0) ** 2
    for st in sites:
        if st is None:
            out.append(None); continue
        sel, meth = st; c = sel.mean(0)
        start = [int(round(x)) - S // 2 for x in c]
        h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0)
        site = np.zeros((S, S, S), bool); q = sel - np.array(start)
        q = q[((q >= 0) & (q < S)).all(1)]; site[tuple(q.T)] = True
        Le = ndimage.binary_erosion(mc == LIV, BALL1, iterations=1)
        out.append((h, np.stack([site & Le, Le & (d2 <= 10 ** 2), Le]), c, meth))
    return out


UNITS = {"gs": unit_gs, "ath": unit_ath, "fst": unit_fst}


def one(args):
    unit, sid, pairs, odir = args
    fo = f"{odir}/{sid}.npz"
    if os.path.exists(fo):
        return sid, "skip", None
    src = next(((p, mk) for p, mk in pairs if os.path.exists(f"{p}/{sid}.npz") and os.path.exists(f"{mk}/{sid}.nii.gz")), None)
    if src is None:
        return sid, "nofile", None
    try:
        z = np.load(f"{src[0]}/{sid}.npz"); hu = z["hu"]
        m = np.asarray(nib.load(f"{src[1]}/{sid}.nii.gz").dataobj).astype(np.uint8).T
    except Exception as e:  # noqa: BLE001
        return sid, "readerr", repr(e)[:80]
    if m.shape != hu.shape:
        return sid, "shape", None
    C = CFG[unit]; S, K, Rn = C["S"], C["K"], C["R"]
    try:
        res = UNITS[unit](sid, z, hu, m, S)
    except Exception as e:  # noqa: BLE001
        return sid, "err", repr(e)[:120]
    H = np.full((K, S, S, S), -1024, np.int16); R = np.zeros((K, Rn, S, S, S), bool)
    pres = np.zeros(K, bool); cent = np.full((K, 3), np.nan, np.float32); meth = []
    for k, r in enumerate(res):
        if r is None:
            meth.append("absent"); continue
        h, Rk, c, mt = r
        H[k] = np.clip(h, -1024, 3071).astype(np.int16); R[k] = Rk; pres[k] = Rk.any(); cent[k] = c; meth.append(mt)
    np.savez_compressed(fo + ".tmp.npz", present=pres, hu=H, regions=np.packbits(R.ravel()), rshape=np.array(R.shape),
                        center=cent, method=np.array(meth), hu_feats=hu_feats(H, R), orient=ORIENT)
    os.replace(fo + ".tmp.npz", fo)
    return sid, "|".join(meth), {"region_vox": R.sum((2, 3, 4)).tolist()}


if __name__ == "__main__":
    unit, ids, odir, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    pairs = [tuple(p.split(":")) for p in sys.argv[5:]]
    os.makedirs(odir, exist_ok=True)
    from collections import Counter
    cnt = Counter()
    L_ = [l.strip() for l in open(ids) if l.strip()]
    log = open(f"{odir.rstrip('/')}_info.jsonl", "a")
    with Pool(npr) as p:
        for i, (sid, st, info) in enumerate(p.imap_unordered(one, [(unit, s, pairs, odir) for s in L_], chunksize=2)):
            cnt[st] += 1
            if st != "skip":
                log.write(json.dumps({"study_id": sid, "status": st, **(info if isinstance(info, dict) else {"msg": info})}, default=float) + "\n")
            if (i + 1) % 200 == 0:
                log.flush(); print(i + 1, len(L_), dict(cnt.most_common(8)), flush=True)
    log.flush()
    print("CROP_DONE", dict(cnt), flush=True)
