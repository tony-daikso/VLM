"""Exp O region crop units (round 2), stage A (CPU). Same pipeline / output format as Exp M ru_crop.py / Exp N nu_crop.py
(read by ru_qc.py and nu_extract.py). Input: 3 mm prep npz (hu z,y,x; RAS: +z superior, +y anterior, +x patient right)
+ TS 'total' 3 mm label map (nii, transposed to z,y,x). Per case the LR/AP/SI signs are checked from the TS
labels (some prep volumes are mirrored); method suffix '_flip<R|A|S>' for non-RAS cases.
Units (--unit):
  bd  : distal CBD / pancreatic head, K=1 crop 48^3 (144 mm). Centre = pancreas voxels within 15 mm of the duodenum
        ('head_duod'); fallback right-most 30 % of the pancreas ('head_right'); fallback duodenum centroid ('duod').
        regions: r0 pancreatic head (pancreas within 30 mm of the centre); r1 head + unlabelled soft tissue within 24 mm
                 (non-bone, non-vessel, HU > -200) = intrapancreatic / distal CBD course; r2 periampullary: duodenum within
                 15 mm of the pancreas, dilated 1 vox.
  fa2 : free-air pockets round 2, K=3 crops 64^3 (192 mm):
        c0 'falciform' = anterior left lobe: liver voxels left of (spine midline + 15 mm) in the upper 60 % of the liver,
           anterior-most 25 % of those -> centroid moved 15 mm anterior;
        c1 'antrum_duod' = midpoint of the antrum (stomach within 20 mm of the duodenum) and the duodenal bulb (duodenum
           within 20 mm of the stomach) (perforation sites); fallback duodenum centroid;
        c2 'rparacolic' = ascending colon (colon voxels with x >= 80th pct of the colon x, below the liver centroid)
           centroid moved 18 mm laterally (right).
        regions (pool over NON-ORGAN voxels): r0 body unlabelled non-bone non-lung; r1 free-air candidates (body, HU < -400,
                 unlabelled, >= 6 mm from GI tract, >= 3 mm from lung/airway); r2 unlabelled body within 45 mm of the centre.
  asc2: ascites pockets round 2, K=3 crops 64^3: c0 'rgutter' = ascending colon centroid moved 18 mm right;
        c1 'lgutter' = descending colon (colon x <= 20th pct, below the spleen centroid) moved 18 mm left;
        c2 'lessersac' = midpoint of the posterior stomach wall (stomach y <= 25th pct) and the pancreas body/tail
           (pancreas x <= median) ; fallback stomach centroid moved 30 mm posterior.
        regions (= Exp M asc): r0 ball 30 mm (body, non-bone); r1 fluid candidates (unlabelled, body, -20..40 HU, <= 60 mm);
                 r2 organ shell (12 mm around colon / stomach+pancreas, unlabelled, body); r3 ball 50 mm (body, non-bone).
  frp : pelvic ring fracture sites, K=4 crops 48^3: c0 right / c1 left acetabulum (hip voxels within 15 mm of the femur of
        the same side; fallback lowest 30 % of the hip bone), c2 pubic rami / symphysis (anterior-most 30 % of both hip bones
        in their lowest 30 %), c3 sacrum incl. sacral ala / SI joints (sacrum centroid).
        regions: r0 site bone (hip / pubic hip voxels / sacrum, + femur head for acetabula) dilated 1 vox; r1 all bone in
                 the crop dilated 1 vox.  (max pooling comes from nu_extract.py mean|max)
  go  : gastric outlet / duodenum, K=2 crops 64^3: c0 'antrum' (stomach within 20 mm of the duodenum; fallback right-most
        25 % of the stomach), c1 'duodenum' (duodenum centroid).
        regions: r0 lumen = stomach | duodenum; r1 site organ (antrum part of the stomach / duodenum) dilated 1 vox;
                 r2 body non-bone within 45 mm.
Per crop scalar HU features hu_feats [K,R,5]: fluid frac (-10..30), air frac (< -400), median, p90, n >= 300 HU.
Output out_dir/{id}.npz: present [K], hu int16 [K,S,S,S], regions packbits of bool [K,R,S,S,S], rshape, center [K,3],
method [K], hu_feats, orient (lr=2, sr, ap=1, sa=1, ss=1).
Usage: no_crop.py UNIT ids.txt out_dir nproc prep1:mask1 [prep2:mask2 ...]   (ids = train/val only)"""
import sys, os, json, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
from totalsegmentator.map_to_binary import class_map

CM = class_map["total"]; INV = {v: k for k, v in CM.items()}
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE_IDS = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
GI_IDS = ids_of("stomach", "duodenum", "small_bowel", "colon", "esophagus")
LUNG_IDS = ids_of("lung_", "trachea")
SPINE_IDS = ids_of("vertebrae_L", "vertebrae_T1")
LIV, SPL, PAN, STO, DUO, COL = INV["liver"], INV["spleen"], INV["pancreas"], INV["stomach"], INV["duodenum"], INV["colon"]
HIPR, HIPL, FEMR, FEML, SAC = INV["hip_right"], INV["hip_left"], INV["femur_right"], INV["femur_left"], INV["sacrum"]
BONE = np.zeros(256, bool); BONE[BONE_IDS] = True
VESSM = np.zeros(256, bool); VESSM[VESS] = True
GI = np.zeros(256, bool); GI[GI_IDS] = True
LUNGT = np.zeros(256, bool); LUNGT[LUNG_IDS] = True
BALL1 = ndimage.generate_binary_structure(3, 1)
CFG = {"bd": dict(S=48, K=1, R=3), "fa2": dict(S=64, K=3, R=3), "asc2": dict(S=64, K=3, R=4),
       "frp": dict(S=48, K=4, R=2), "go": dict(S=64, K=2, R=3)}


def crop(a, start, S, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    if all(d.stop > d.start for d in dst):
        out[dst] = a[src]
    return out


def body_mask(hu):
    b = hu > -500
    lab, n = ndimage.label(b)
    if n > 1:
        sz = np.bincount(lab.ravel()); sz[0] = 0; b = lab == sz.argmax()
    return np.stack([ndimage.binary_fill_holes(s) for s in b])


def d2grid(S):
    zz, yy, xx = np.indices((S, S, S)); c0 = S // 2
    return (zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2


def near(A, B, vox):
    """voxels of A within `vox` voxels of B (EDT on the joint bounding box)"""
    if not A.any() or not B.any():
        return np.zeros_like(A)
    U = A | B; nz = np.argwhere(U); lo = np.maximum(nz.min(0) - vox - 1, 0); hi = nz.max(0) + vox + 2
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    d = ndimage.distance_transform_edt(~B[sl])
    out = np.zeros_like(A); out[sl] = A[sl] & (d <= vox)
    return out


def hu_feats(h, R):
    K, Rn = R.shape[:2]; hf = np.full((K, Rn, 5), np.nan, np.float32)
    for k in range(K):
        for r in range(Rn):
            v = h[k][R[k, r]].astype(np.float32)
            if v.size < 10:
                continue
            hf[k, r] = [((v >= -10) & (v <= 30)).mean(), (v < -400).mean(), np.median(v), np.percentile(v, 90), (v >= 300).sum()]
    return hf


def lr_sign(m):
    a, b = np.argwhere(m == LIV), np.argwhere(m == SPL)
    if len(a) >= 1000 and len(b) >= 200:
        return 1 if a[:, 2].mean() > b[:, 2].mean() else -1
    return 1


def orient_signs(m):
    """per-case signs (+1 = RAS prep convention): sr (x toward patient right), sa (y anterior), ss (z superior).
    The prep affine is not always trustworthy (some cases are mirrored): LR from liver vs spleen, AP from spine vs
    small bowel/colon, SI from liver vs bladder/sacrum."""
    sr = lr_sign(m)
    sp = np.argwhere(np.isin(m, SPINE_IDS)); bw = np.argwhere(np.isin(m, [INV["small_bowel"], COL]))
    sa = (1 if bw[:, 1].mean() > sp[:, 1].mean() else -1) if len(sp) >= 50 and len(bw) >= 500 else 1
    L = np.argwhere(m == LIV); lo = np.argwhere(np.isin(m, [INV["urinary_bladder"], SAC]))
    ss = (1 if L[:, 0].mean() > lo[:, 0].mean() else -1) if len(L) >= 1000 and len(lo) >= 300 else 1
    return sr, sa, ss


def mk_crop(hu, m, c, S, body=None):
    start = [int(round(x)) - S // 2 for x in c]
    h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0)
    bd = crop(body, start, S, False) if body is not None else None
    return start, h, mc, bd


# ------------------------------------------------------------------ distal CBD / pancreatic head
def unit_bd(hu, m, o, S):
    sr, sa, ss = o
    P, D = m == PAN, m == DUO
    cen = None
    if P.sum() >= 100:
        hd = near(P, D, 5)
        if hd.sum() >= 30:
            cen = (np.argwhere(hd).mean(0), "head_duod")
        else:
            pz = np.argwhere(P); xr = pz[:, 2] * sr
            cen = (pz[xr >= np.percentile(xr, 70)].mean(0), "head_right")
    elif D.sum() >= 100:
        cen = (np.argwhere(D).mean(0), "duod")
    if cen is None:
        return [None]
    c, meth = cen
    start, h, mc, _ = mk_crop(hu, m, c, S)
    d2 = d2grid(S)
    Pc, Dc = mc == PAN, mc == DUO
    r0 = Pc & (d2 <= 10 ** 2)
    r1 = (Pc | (mc == 0)) & ~BONE[mc] & ~VESSM[mc] & (h > -200) & (d2 <= 8 ** 2)
    r2 = ndimage.binary_dilation(near(Dc, Pc, 5), BALL1) & ~BONE[mc] if Pc.any() and Dc.any() else np.zeros_like(Pc)
    return [(h, np.stack([r0, r1, r2]), c, meth)]


# ------------------------------------------------------------------ free air round 2
def fa_regions(h, mc, bd, S):
    d2 = d2grid(S)
    lungd = ndimage.binary_dilation(LUNGT[mc], BALL1)
    gid = ndimage.binary_dilation(GI[mc], BALL1, iterations=2)
    r0 = bd & (mc == 0) & ~lungd
    r1 = bd & (h < -400) & (mc == 0) & ~gid & ~lungd
    r2 = bd & (mc == 0) & (d2 <= 15 ** 2) & ~lungd
    return np.stack([r0, r1, r2])


def asc_centres(m, sr, ss):
    out = {}
    C = np.argwhere(m == COL)
    if len(C) >= 500:
        xr = C[:, 2] * sr; zc = C[:, 0] * ss
        L = np.argwhere(m == LIV); zl = (L[:, 0] * ss).mean() if len(L) >= 1000 else np.inf
        a = C[(xr >= np.percentile(xr, 80)) & (zc <= zl)]
        if len(a) >= 50:
            c = a.mean(0); c[2] += sr * 6; out["r"] = (c, "asc_colon")
        Sp = np.argwhere(m == SPL); zs = (Sp[:, 0] * ss).mean() if len(Sp) >= 200 else np.inf
        b = C[(xr <= np.percentile(xr, 20)) & (zc <= zs)]
        if len(b) >= 50:
            c = b.mean(0); c[2] -= sr * 6; out["l"] = (c, "desc_colon")
    return out


def unit_fa2(hu, m, o, S):
    sr, sa, ss = o
    body = body_mask(hu)
    cents = []
    L = np.argwhere(m == LIV); sp = np.argwhere(np.isin(m, SPINE_IDS))
    if len(L) >= 1000:
        mid = np.median(sp[:, 2]) if len(sp) >= 50 else np.median(L[:, 2])
        z = L[:, 0] * ss; sel = L[((L[:, 2] - mid) * sr <= 5) & (z >= np.percentile(z, 40))]
        if len(sel) >= 50:
            ya = sel[:, 1] * sa; top = sel[ya >= np.percentile(ya, 75)]
            c = top.mean(0); c[1] += sa * 5; cents.append((c, "falciform"))
        else:
            cents.append(None)
    else:
        cents.append(None)
    S_, D = m == STO, m == DUO
    if S_.sum() >= 200 and D.sum() >= 100:
        an, bu = near(S_, D, 7), near(D, S_, 7)
        if an.sum() >= 20 and bu.sum() >= 20:
            cents.append(((np.argwhere(an).mean(0) + np.argwhere(bu).mean(0)) / 2, "antrum_duod"))
        else:
            cents.append((np.argwhere(D).mean(0), "duod"))
    elif D.sum() >= 100:
        cents.append((np.argwhere(D).mean(0), "duod"))
    else:
        cents.append(None)
    ac = asc_centres(m, sr, ss)
    cents.append(ac.get("r"))
    out = []
    for cc in cents:
        if cc is None:
            out.append(None); continue
        c, meth = cc
        start, h, mc, bd = mk_crop(hu, m, c, S, body)
        out.append((h, fa_regions(h, mc, bd, S), c, meth))
    return out


# ------------------------------------------------------------------ ascites round 2
def unit_asc2(hu, m, o, S):
    sr, sa, ss = o
    body = body_mask(hu)
    ac = asc_centres(m, sr, ss)
    cents = [ac.get("r") and (*ac["r"], [COL]), ac.get("l") and (*ac["l"], [COL])]
    St = np.argwhere(m == STO); Pz = np.argwhere(m == PAN)
    if len(St) >= 200 and len(Pz) >= 100:
        ya = St[:, 1] * sa; post = St[ya <= np.percentile(ya, 25)].mean(0)
        xr = Pz[:, 2] * sr; body_tail = Pz[xr <= np.median(xr)].mean(0)
        cents.append(((post + body_tail) / 2, "lessersac", [STO, PAN]))
    elif len(St) >= 200:
        c = St.mean(0); c[1] -= sa * 10; cents.append((c, "stomach_post", [STO]))
    else:
        cents.append(None)
    d2 = d2grid(S)
    out = []
    for cc in cents:
        if not cc:
            out.append(None); continue
        c, meth, org = cc
        start, h, mc, bd = mk_crop(hu, m, c, S, body)
        nb = bd & ~BONE[mc]; om = np.isin(mc, org)
        r0 = nb & (d2 <= 10 ** 2)
        r1 = bd & (mc == 0) & (h >= -20) & (h <= 40) & (d2 <= 20 ** 2)
        r2 = (ndimage.binary_dilation(om, BALL1, iterations=4) & ~om & (mc == 0) & bd) if om.any() else np.zeros_like(om)
        r3 = nb & (d2 <= 17 ** 2)
        out.append((h, np.stack([r0, r1, r2, r3]), c, meth))
    return out


# ------------------------------------------------------------------ pelvic ring
def unit_frp(hu, m, o, S):
    sr, sa, ss = o
    cents = []
    for hp, fm, nm in ((HIPR, FEMR, "acetab_r"), (HIPL, FEML, "acetab_l")):
        H, Fm = m == hp, m == fm
        if H.sum() < 300:
            cents.append(None); continue
        ac = near(H, Fm, 5) if Fm.sum() >= 100 else np.zeros_like(H)
        if ac.sum() >= 30:
            cents.append((np.argwhere(ac).mean(0), nm, [hp, fm]))
        else:
            hz = np.argwhere(H); zz = hz[:, 0] * ss; cents.append((hz[zz <= np.percentile(zz, 30)].mean(0), nm + "_low", [hp]))
    Hb = np.argwhere(np.isin(m, [HIPR, HIPL]))
    if len(Hb) >= 600:
        zz = Hb[:, 0] * ss; low = Hb[zz <= np.percentile(zz, 30)]
        ya = low[:, 1] * sa; ant = low[ya >= np.percentile(ya, 70)]
        cents.append((ant.mean(0), "pubic", [HIPR, HIPL]) if len(ant) >= 30 else None)
    else:
        cents.append(None)
    Sc = np.argwhere(m == SAC)
    cents.append((Sc.mean(0), "sacrum", [SAC]) if len(Sc) >= 300 else None)
    out = []
    for k, cc in enumerate(cents):
        if cc is None:
            out.append(None); continue
        c, meth, labs = cc
        start, h, mc, _ = mk_crop(hu, m, c, S)
        site = np.isin(mc, labs)
        if k == 2:                               # pubic: only the anterior-inferior part of the hip bones in the crop
            site &= ((np.indices((S, S, S))[0] - S // 2) * ss <= 8)
        r0 = ndimage.binary_dilation(site, BALL1)
        r1 = ndimage.binary_dilation(BONE[mc], BALL1)
        out.append((h, np.stack([r0, r1]), c, meth))
    return out


# ------------------------------------------------------------------ gastric outlet / duodenum
def unit_go(hu, m, o, S):
    sr, sa, ss = o
    body = body_mask(hu)
    S_, D = m == STO, m == DUO
    cents = []
    if S_.sum() >= 200:
        an = near(S_, D, 7) if D.sum() >= 100 else np.zeros_like(S_)
        if an.sum() >= 20:
            cents.append((np.argwhere(an).mean(0), "antrum", STO))
        else:
            sz = np.argwhere(S_); xr = sz[:, 2] * sr
            cents.append((sz[xr >= np.percentile(xr, 75)].mean(0), "stomach_right", STO))
    else:
        cents.append(None)
    cents.append((np.argwhere(D).mean(0), "duodenum", DUO) if D.sum() >= 100 else None)
    d2 = d2grid(S)
    out = []
    for cc in cents:
        if cc is None:
            out.append(None); continue
        c, meth, lab = cc
        start, h, mc, bd = mk_crop(hu, m, c, S, body)
        lum = np.isin(mc, [STO, DUO])
        site = mc == lab
        if lab == STO and (mc == DUO).any():
            site = near(site, mc == DUO, 7)
            if site.sum() < 20:
                site = mc == STO
        r1 = ndimage.binary_dilation(site, BALL1)
        r2 = bd & ~BONE[mc] & (d2 <= 15 ** 2)
        out.append((h, np.stack([lum, r1, r2]), c, meth))
    return out


UNITS = {"bd": unit_bd, "fa2": unit_fa2, "asc2": unit_asc2, "frp": unit_frp, "go": unit_go}


def one(args):
    unit, sid, pairs, odir = args
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
    C = CFG[unit]; S, K, Rn = C["S"], C["K"], C["R"]
    o = orient_signs(m); sr = o[0]
    try:
        res = UNITS[unit](hu, m, o, S)
    except Exception as e:  # noqa: BLE001
        return sid, "err", repr(e)[:120]
    H = np.full((K, S, S, S), -1024, np.int16); R = np.zeros((K, Rn, S, S, S), bool)
    pres = np.zeros(K, bool); cent = np.full((K, 3), np.nan, np.float32); meth = []
    for k, r in enumerate(res):
        if r is None:
            meth.append("absent"); continue
        h, Rk, c, mt = r
        H[k] = np.clip(h, -1024, 3071).astype(np.int16); R[k] = Rk; pres[k] = Rk.any(); cent[k] = c
        meth.append(mt + ("" if o == (1, 1, 1) else "_flip" + "".join("RAS"[i] for i in range(3) if o[i] < 0)))
    np.savez_compressed(fo + ".tmp.npz", present=pres, hu=H, regions=np.packbits(R.ravel()), rshape=np.array(R.shape),
                        center=cent, method=np.array(meth), hu_feats=hu_feats(H, R), orient=np.array([2, o[0], 1, o[1], o[2]]))
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
