"""RLQ / cecal crop unit, stage A (CPU). Mirrors biliary bd5_crop.py.
Input : 3 mm prep npz (hu, z,y,x) + TotalSegmentator 'total' 3 mm mask (nii, transposed to z,y,x).
Orientation per case (no affine assumptions):
  SI  axis/sign: axis 0 (z); superior = side of liver vs urinary bladder / sacrum.
  AP  axis/sign: of the two in-plane axes, the one with the larger lumbar-spine -> small-bowel offset; anterior = bowel side.
  LR  axis      : the other in-plane axis; patient right = liver side relative to spine (fallback: away from spleen).
Cecum / ileocecal (ICV) heuristic:
  1. midline = LR centroid of vertebrae L1-S1 + sacrum.  Right colon RC = colon voxels >= 15 mm right of midline.
  2. cecal band B = RC voxels within 60 mm above the caudal tip of RC (tip = 2nd percentile of SI coordinate).
  3. iliopsoas_right (TS 89) in the same SI band: keep B voxels that are anterior to the iliopsoas centroid OR medial to
     its centroid (anterior/medial constraint, drops colon posterolateral to the muscle). Skipped if no iliopsoas.
  4. ICV proxy = constrained B voxels within 6 mm of small_bowel (TS 18).  >=10 voxels -> centre = their median
     (method 'icv'); else centre = centroid of constrained B (method 'cectip'); no RC -> iliopsoas_right anterior
     surface at its lower third (method 'psoas'); otherwise present=False.
Crop 64^3 voxels (192 mm) around the centre, pad -1024 / 0.
Region masks (bool, may overlap) for mask pooling in stage B:
  r0 ball30   : all voxels within 30 mm of the centre
  r1 pericecal: unlabelled body voxels (mask 0, HU > -500) within 24 mm of the cecal band B (appendix + mesoappendix fat)
  r2 cecum_ti : cecal band B (colon) + small bowel within 30 mm of B (cecal wall/lumen + terminal ileum)
  r3 rlq50    : all non-bone voxels within 50 mm of the centre
Also scalar HU features per region (fat-stranding fraction channel): strand_frac (-70..20 among -190..60),
fluid_frac (0..30), fat_med (-190..-30 median) -> hu_feats [4,3].
Usage: rlq_crop.py ids.txt out_dir nproc prep1:mask1 [prep2:mask2 ...]   (ids = train/val only)."""
import sys, os, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
S = 64
COLON, SB, IPR, SPLEEN, LIVER, BLADDER = 20, 18, 89, 1, 5, 21
SPINE = [25, 26, 27, 28, 29, 30, 31]
BONE = np.zeros(256, bool); BONE[25:51] = True; BONE[69:79] = True; BONE[91:118] = True


def crop(a, start, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = a[src]
    return out


def cen(m, labs):
    a = np.argwhere(np.isin(m, labs))
    return (a.mean(0), len(a)) if len(a) else (None, 0)


def orient(m):
    """returns (lr_axis, sgn_right, ap_axis, sgn_ant, sgn_sup) for z,y,x arrays; None if impossible."""
    cs, ns = cen(m, SPINE); cb, nb = cen(m, [SB, COLON]); cl, nl = cen(m, [LIVER])
    if cs is None or cb is None or nb < 500:
        return None
    off = np.abs((cb - cs)[1:]); ap = 1 + int(np.argmax(off)); lr = 3 - ap
    sgn_ant = 1 if cb[ap] > cs[ap] else -1
    if cl is not None and nl > 500:
        sgn_r = 1 if cl[lr] > cs[lr] else -1
    else:
        cp, npn = cen(m, [SPLEEN])
        if cp is None:
            return None
        sgn_r = -1 if cp[lr] > cs[lr] else 1
    ref, nr = cen(m, [BLADDER, 25])
    if cl is not None and ref is not None:
        sgn_s = 1 if cl[0] > ref[0] else -1
    else:
        sgn_s = 1 if (cl is not None and cl[0] > cs[0]) else -1
    return lr, sgn_r, ap, sgn_ant, sgn_s


def locate(m):
    """returns (centre, method, info, band_coords)"""
    o = orient(m)
    if o is None:
        return None, "noorient", {}, None
    lr, sr, ap, sa, ss = o
    info = dict(lr_axis=lr, sgn_right=sr, ap_axis=ap, sgn_ant=sa, sgn_sup=ss)
    cs, _ = cen(m, SPINE); mid = cs[lr]
    colm = m == COLON
    ip = np.argwhere(m == IPR)
    # (a) ascending-colon seed: colon >= 40 mm right of midline at the L1-L4 level
    zl = [cen(m, [k])[0] for k in (28, 31)]                        # L4, L1 centroids
    rightmap = lambda c: (c[:, lr] - mid) * sr * 3.0
    col = np.argwhere(colm); rc = rightmap(col)
    comp_ok = None
    if zl[0] is not None and zl[1] is not None:
        z0, z1 = sorted([zl[0][0], zl[1][0]])
        seed = (rc >= 40) & (col[:, 0] >= z0) & (col[:, 0] <= z1)
        if seed.sum() >= 50:
            half = colm.copy()
            shp = [1, 1, 1]; shp[lr] = m.shape[lr]; idx = np.arange(m.shape[lr]).reshape(shp)
            half &= ((idx - mid) * sr * 3.0 >= 0)
            lab, n = ndimage.label(half, structure=np.ones((3, 3, 3)))
            ls = np.unique(lab[tuple(col[seed].T)]); ls = ls[ls > 0]
            comp_ok = np.isin(lab[tuple(col.T)], ls)
            info["seed_vox"] = int(seed.sum())
    keep = rc >= 25
    if comp_ok is not None and (keep & comp_ok).sum() >= 100:
        keep &= comp_ok; meth_pre = "asc"
    else:
        meth_pre = "noseed"
    # (b) drop rectum/sigmoid near the sacrum (<= 20 mm from sacrum/S1)
    sac = np.isin(m, [25, 26])
    if sac.any():
        lo_ = np.maximum(col[keep].min(0) - 8, 0) if keep.any() else np.zeros(3, int)
        hi_ = np.minimum(col[keep].max(0) + 9, m.shape) if keep.any() else np.array(m.shape)
        sl_ = tuple(slice(a, b) for a, b in zip(lo_, hi_))
        if sac[sl_].any():
            dsac = ndimage.distance_transform_edt(~sac[sl_])
            q = col[keep] - lo_
            near = dsac[tuple(q.T)] <= 20 / 3.0
            kk = np.where(keep)[0]; keep[kk[near]] = False
    RC = col[keep]
    info["method_pre"] = meth_pre
    if len(RC) < 100:
        if len(ip) > 200:
            sup = ip[:, 0] * ss; low = ip[sup <= np.percentile(sup, 33)]
            ant = low[:, ap] * sa; c = low[ant >= np.percentile(ant, 90)].mean(0)
            return c, "psoas", info, None
        return None, "nocolon", info, None
    sup = RC[:, 0] * ss; tip = np.percentile(sup, 2)
    B = RC[sup <= tip + 20]                                    # 60 mm band above the caudal tip
    info["band_vox"] = len(B)
    Bc = B
    # (c) anterior/medial to iliopsoas_right: keep band voxels anterior to the muscle centroid (-5 mm tolerance) or lateral to it
    #     (cecum lies anterior / anteromedial to the iliopsoas; colon posterior AND medial to it = rectosigmoid)
    if len(ip) > 100:
        ips = ip[:, 0] * ss; bs = B[:, 0] * ss
        ipb = ip[(ips >= bs.min() - 5) & (ips <= bs.max() + 5)]
        if len(ipb) > 50:
            ipc = ipb.mean(0)
            ant = (B[:, ap] - ipc[ap]) * sa; med = (B[:, lr] - ipc[lr]) * sr < 0
            ok = (ant > -5) | ~med
            info["ip_keep_frac"] = float(ok.mean())
            if ok.sum() >= 20:
                Bc = B[ok]
    lo = np.maximum(Bc.min(0) - 4, 0); hi = np.minimum(Bc.max(0) + 5, m.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sbm = m[sl] == SB
    if sbm.any():
        d = ndimage.distance_transform_edt(~sbm)
        icv = Bc[d[tuple((Bc - lo).T)] <= 2]
    else:
        icv = Bc[:0]
    info["icv_vox"] = len(icv)
    if len(icv) >= 10:
        return np.median(icv, 0), "icv", info, Bc
    return Bc.mean(0), "cectip", info, Bc


def one(args):
    sid, pairs, odir = args
    fo = f"{odir}/{sid}.npz"
    if os.path.exists(fo):
        return sid, "skip", None
    src = next(((p, mk) for p, mk in pairs if os.path.exists(f"{p}/{sid}.npz") and os.path.exists(f"{mk}/{sid}.nii.gz")), None)
    if src is None:
        return sid, "nofile", None
    try:
        hu = np.load(f"{src[0]}/{sid}.npz")["hu"]
        m = np.asarray(nib.load(f"{src[1]}/{sid}.nii.gz").dataobj).astype(np.uint8).T
    except Exception as e:  # noqa: BLE001
        return sid, "readerr", repr(e)[:80]
    if m.shape != hu.shape:
        return sid, "shape", None
    c, meth, info, Bc = locate(m)
    if c is None:
        np.savez_compressed(fo + ".tmp.npz", present=False, method=meth); os.replace(fo + ".tmp.npz", fo)
        return sid, meth, info
    start = [int(round(x)) - S // 2 for x in c]
    h = crop(hu, start, -1024); mc = crop(m, start, 0)
    zz, yy, xx = np.indices((S, S, S)); c0 = S // 2
    d2 = (zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2
    r0 = d2 <= 10 ** 2
    lr, sr, ap, sa, ss = (info[k] for k in ("lr_axis", "sgn_right", "ap_axis", "sgn_ant", "sgn_sup"))
    # cecal band inside the crop (constrained band voxels from locate)
    band = np.zeros((S, S, S), bool)
    if Bc is not None:
        q = Bc - np.array(start); q = q[((q >= 0) & (q < S)).all(1)]
        band[tuple(q.T)] = True
    if not band.any():
        band = (mc == COLON) & (d2 <= 13 ** 2)
    db = ndimage.distance_transform_edt(~band) if band.any() else np.full(h.shape, 99.)
    free = (mc == 0) & (h > -500)
    r1 = free & (db <= 8)
    r2 = band | ((mc == SB) & (db <= 10))
    r3 = (d2 <= 17 ** 2) & ~BONE[mc]
    R = np.stack([r0, r1, r2, r3])
    hf = np.full((4, 3), np.nan, np.float32)
    for k in range(4):
        v = h[R[k]].astype(np.float32)
        if v.size < 20:
            continue
        soft = v[(v >= -190) & (v <= 60)]; fat = v[(v >= -190) & (v <= -30)]
        hf[k, 0] = ((soft >= -70) & (soft <= 20)).mean() if soft.size else np.nan
        hf[k, 1] = ((v >= 0) & (v <= 30)).mean()
        hf[k, 2] = np.median(fat) if fat.size > 10 else np.nan
    np.savez_compressed(fo + ".tmp.npz", present=True, hu=h.astype(np.int16), regions=R, start=np.array(start),
                        center=np.array(c), method=meth, hu_feats=hf,
                        orient=np.array([lr, sr, ap, sa, ss]))
    os.replace(fo + ".tmp.npz", fo)
    return sid, meth, info


if __name__ == "__main__":
    ids, odir, npr = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pairs = [tuple(p.split(":")) for p in sys.argv[4:]]
    os.makedirs(odir, exist_ok=True)
    from collections import Counter
    cnt = Counter()
    L_ = [l.strip() for l in open(ids) if l.strip()]
    log = open(f"{odir}/../crop_info.jsonl", "a")
    import json
    with Pool(npr) as p:
        for i, (sid, st, info) in enumerate(p.imap_unordered(one, [(s, pairs, odir) for s in L_], chunksize=4)):
            cnt[st] += 1
            if st != "skip":
                log.write(json.dumps({"study_id": sid, "status": st, **(info if isinstance(info, dict) else {"msg": info})},
                                     default=float) + "\n")
            if (i + 1) % 500 == 0:
                log.flush(); print(i + 1, len(L_), dict(cnt), flush=True)
    print("CROP_DONE", dict(cnt), flush=True)
