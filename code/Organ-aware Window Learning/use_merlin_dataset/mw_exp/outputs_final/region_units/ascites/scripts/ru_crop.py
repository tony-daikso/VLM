"""Exp M region crop units, stage A (CPU). Generalises appendicitis rlq_crop.py / biliary bd5_crop.py to findings whose
lesion has no own TS mask. Input: 3 mm prep npz (hu, z,y,x) + TS 'total' 3 mm label map (nii, transposed to z,y,x).
Orientation per case (no affine assumptions; same rules as rlq_crop.orient):
  SI sign: liver vs bladder/sacrum;  AP axis: in-plane axis with the larger spine->bowel offset, anterior = bowel side;
  LR axis: the other one, patient right = liver side (fallback: away from spleen).
Units (--unit):
  hn  : renal sinus / pelvis unit, one 48^3 crop per kidney (K=2: right, left), centred on the renal sinus
        (sinus = 3D closing(kidney, 9 mm) + per-slice hole fill - kidney, unlabelled/cyst voxels, as hn_hu.py) = hilum /
        medial concavity; fallback = kidney centroid moved 15 mm medially (method 'medial').
        regions: r0 sinus (dilated 1 vox inside free space), r1 kidney (+cyst), r2 hilum ball 24 mm (non-bone),
                 r3 perirenal shell (12 mm around kidney, outside closing, free, HU > -500).
  fa  : free-air units, 64^3 (192 mm), K=2:
        c0 'subdiaph' = anterior part of the liver dome (liver voxels in the top 30 % of its SI extent, anterior 30 % of
           those) moved 15 mm anterior;  c1 'antabd' = midline (spine LR) at L4 (fallback L5/L3) level, 60 mm deep to the
           anterior skin.
        regions: r0 body non-bone non-lung, r1 free-air candidates (body, HU < -400, unlabelled, >= 6 mm from GI tract,
                 >= 3 mm from lung/airway), r2 anterior non-dependent band (body within 30 mm of skin, anterior half of the
                 crop), r3 unlabelled body tissue within 60 mm of the centre (non-bone).
  asc : ascites peritoneal pockets, 64^3, K=3:
        c0 'morison' = midpoint of the right-kidney upper pole and the postero-inferior liver (liver only if no kidney);
        c1 'perisplenic' = spleen centroid moved 15 mm inferior;  c2 'pelvic' = bladder centroid moved 25 mm posterior
           (fallback: lower sacrum moved 50 mm anterior, method 'sacrum').
        regions: r0 ball 30 mm (body, non-bone), r1 fluid candidates (unlabelled, body, -20..40 HU, within 60 mm),
                 r2 organ shell (12 mm around liver+right kidney / spleen / bladder, unlabelled, body), r3 ball 50 mm.
Per crop scalar HU features hu_feats [K,R,3]: fluid frac (-10..30), air frac (< -400), median HU.
Output out_dir/{id}.npz: hu int16 [K,S,S,S], regions packed bits (np.packbits of bool [K,R,S,S,S]), present bool [K],
center [K,3], method [K] str, hu_feats.  Usage: ru_crop.py UNIT ids.txt out_dir nproc prep1:mask1 [prep2:mask2 ...]
(ids = train/val only)."""
import sys, os, json, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool

COLON, SB, SPLEEN, LIVER, BLADDER, STOMACH, DUOD, ESO, TRACH = 20, 18, 1, 5, 21, 6, 19, 15, 16
KR, KL, CYST_R, CYST_L = 2, 3, 24, 23
SPINE = [25, 26, 27, 28, 29, 30, 31]
LUNG = [10, 11, 12, 13, 14]
BONE = np.zeros(256, bool); BONE[25:51] = True; BONE[69:79] = True; BONE[91:118] = True
LUNGT = np.zeros(256, bool); LUNGT[LUNG] = True; LUNGT[TRACH] = True
GI = np.zeros(256, bool); GI[[STOMACH, SB, DUOD, COLON, ESO]] = True
BALL1 = ndimage.generate_binary_structure(3, 1)
CFG = {"hn": dict(S=48, K=2, R=4, names=["right", "left"]),
       "fa": dict(S=64, K=2, R=4, names=["subdiaph", "antabd"]),
       "asc": dict(S=64, K=3, R=4, names=["morison", "perisplenic", "pelvic"])}


def crop(a, start, S, pad):
    out = np.full((S, S, S), pad, a.dtype)
    src = tuple(slice(max(s, 0), min(s + S, n)) for s, n in zip(start, a.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    if all(d.stop > d.start for d in dst):
        out[dst] = a[src]
    return out


def cen(m, labs):
    a = np.argwhere(np.isin(m, labs))
    return (a.mean(0), len(a)) if len(a) else (None, 0)


def orient(m):
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


def body_mask(hu):
    b = hu > -500
    lab, n = ndimage.label(b)
    if n > 1:
        sz = np.bincount(lab.ravel()); sz[0] = 0; b = lab == sz.argmax()
    return np.stack([ndimage.binary_fill_holes(s) for s in b])


def grid(S):
    zz, yy, xx = np.indices((S, S, S)); c0 = S // 2
    return np.stack([zz, yy, xx]), (zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2


def hu_feats(h, R):
    K, Rn = R.shape[:2]; hf = np.full((K, Rn, 3), np.nan, np.float32)
    for k in range(K):
        for r in range(Rn):
            v = h[k][R[k, r]].astype(np.float32)
            if v.size < 10:
                continue
            hf[k, r] = [((v >= -10) & (v <= 30)).mean(), (v < -400).mean(), np.median(v)]
    return hf


# ------------------------------------------------------------------ hydronephrosis (renal sinus / pelvis)
def unit_hn(hu, m, o, S):
    lr, sr, ap, sa, ss = o
    cs, _ = cen(m, SPINE)
    G, d2 = grid(S)
    out = []
    for kl, cy in ((KR, CYST_R), (KL, CYST_L)):
        K = m == kl
        if K.sum() < 300:
            out.append(None); continue
        kk = np.argwhere(K); kc0 = kk.mean(0)
        lo = np.maximum(kk.min(0) - 6, 0); hi = np.minimum(kk.max(0) + 7, m.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        k = K[sl]; L = m[sl]
        kc = ndimage.binary_closing(k, BALL1, iterations=3)
        kc = np.stack([ndimage.binary_fill_holes(a) for a in kc]) | k
        free = (L == 0) | (L == cy)
        sinus = kc & ~k & free
        if sinus.sum() >= 20:
            c = np.argwhere(sinus).mean(0) + lo; meth = "sinus"
        else:
            c = kc0.copy(); c[lr] += np.sign(cs[lr] - kc0[lr]) * 5; meth = "medial"
        start = [int(round(x)) - S // 2 for x in c]
        h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0)
        # recompute closing inside the crop (kidney fits in 144 mm)
        kk_ = mc == kl
        kcc = ndimage.binary_closing(kk_, BALL1, iterations=3)
        kcc = np.stack([ndimage.binary_fill_holes(a) for a in kcc]) | kk_
        fr = ((mc == 0) | (mc == cy)) & (h > -500)
        sn = kcc & ~kk_ & fr
        r0 = (ndimage.binary_dilation(sn, BALL1) & fr) if sn.any() else (d2 <= 3 ** 2) & fr
        r1 = kk_ | (mc == cy)
        r2 = (d2 <= 8 ** 2) & ~BONE[mc]
        r3 = ndimage.binary_dilation(kk_, BALL1, iterations=4) & ~kcc & fr
        out.append((h, np.stack([r0, r1, r2, r3]), c, meth))
    return out


# ------------------------------------------------------------------ free air
def unit_fa(hu, m, o, S):
    lr, sr, ap, sa, ss = o
    G, d2 = grid(S); c0 = S // 2
    body = body_mask(hu)
    out = []
    # c0 subdiaphragmatic / anterior perihepatic
    L = np.argwhere(m == LIVER)
    if len(L) >= 1000:
        s = L[:, 0] * ss; top = L[s >= np.percentile(s, 70)]
        a = top[:, ap] * sa; sel = top[a >= np.percentile(a, 70)]
        c = sel.mean(0); c[ap] += sa * 5; cen_a = (c, "liverdome")
    else:
        cen_a = None
    # c1 anterior abdomen at L4 level, midline, 60 mm deep to the skin
    cen_b = None
    cs, _ = cen(m, SPINE)
    for lv, nm in ((28, "L4"), (27, "L5"), (29, "L3")):
        cv, nv = cen(m, [lv])
        if cv is not None and nv > 50:
            z = int(round(cv[0])); mid = int(round(cv[lr]))
            prof = body[z].take(mid, axis=lr - 1)           # 1D along the AP axis
            idx = np.where(prof)[0]
            if len(idx) == 0:
                continue
            ant_edge = idx.max() if sa > 0 else idx.min()
            c = np.array(cv, float); c[ap] = ant_edge - sa * 20
            cen_b = (c, nm); break
    for cc in (cen_a, cen_b):
        if cc is None:
            out.append(None); continue
        c, meth = cc
        start = [int(round(x)) - S // 2 for x in c]
        h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0); bd = crop(body, start, S, False)
        lungd = ndimage.binary_dilation(LUNGT[mc], BALL1)
        gid = ndimage.binary_dilation(GI[mc], BALL1, iterations=2)
        r0 = bd & ~BONE[mc] & ~LUNGT[mc]
        r1 = bd & (h < -400) & (mc == 0) & ~gid & ~lungd
        dsk = ndimage.distance_transform_edt(bd)
        r2 = bd & (dsk <= 10) & ((G[ap] - c0) * sa >= 0) & ~BONE[mc] & ~lungd
        r3 = bd & (mc == 0) & (d2 <= 20 ** 2) & ~lungd
        out.append((h, np.stack([r0, r1, r2, r3]), c, meth))
    return out


# ------------------------------------------------------------------ ascites
def unit_asc(hu, m, o, S):
    lr, sr, ap, sa, ss = o
    G, d2 = grid(S)
    body = body_mask(hu)
    cents = []
    L = np.argwhere(m == LIVER); Kr = np.argwhere(m == KR)
    if len(L) >= 1000:
        s = L[:, 0] * ss; low = L[s <= np.percentile(s, 30)]
        a = low[:, ap] * sa; lp = low[a <= np.percentile(a, 50)].mean(0)
        if len(Kr) >= 300:
            sk = Kr[:, 0] * ss; kt = Kr[sk >= np.percentile(sk, 70)].mean(0)
            cents.append(((lp + kt) / 2, "liver_kidney", [LIVER, KR]))
        else:
            cents.append((lp, "liver", [LIVER]))
    else:
        cents.append(None)
    Sp = np.argwhere(m == SPLEEN)
    if len(Sp) >= 300:
        c = Sp.mean(0); c[0] -= ss * 5; cents.append((c, "spleen", [SPLEEN]))
    else:
        cents.append(None)
    B = np.argwhere(m == BLADDER)
    if len(B) >= 300:
        c = B.mean(0); c[ap] -= sa * 8; cents.append((c, "bladder", [BLADDER]))
    else:
        Sc = np.argwhere(m == 25)
        if len(Sc) >= 300:
            s = Sc[:, 0] * ss; c = Sc[s <= np.percentile(s, 33)].mean(0); c[ap] += sa * 17
            cents.append((c, "sacrum", [COLON]))
        else:
            cents.append(None)
    out = []
    for cc in cents:
        if cc is None:
            out.append(None); continue
        c, meth, org = cc
        start = [int(round(x)) - S // 2 for x in c]
        h = crop(hu, start, S, -1024); mc = crop(m, start, S, 0); bd = crop(body, start, S, False)
        nb = bd & ~BONE[mc]
        org_m = np.isin(mc, org)
        r0 = nb & (d2 <= 10 ** 2)
        r1 = bd & (mc == 0) & (h >= -20) & (h <= 40) & (d2 <= 20 ** 2)
        r2 = (ndimage.binary_dilation(org_m, BALL1, iterations=4) & ~org_m & (mc == 0) & bd) if org_m.any() else r0 & False
        r3 = nb & (d2 <= 17 ** 2)
        out.append((h, np.stack([r0, r1, r2, r3]), c, meth))
    return out


UNITS = {"hn": unit_hn, "fa": unit_fa, "asc": unit_asc}


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
    o = orient(m)
    if o is None:
        np.savez_compressed(fo + ".tmp.npz", present=np.zeros(K, bool), method=np.array(["noorient"] * K)); os.replace(fo + ".tmp.npz", fo)
        return sid, "noorient", None
    res = UNITS[unit](hu, m, o, S)
    H = np.full((K, S, S, S), -1024, np.int16); R = np.zeros((K, Rn, S, S, S), bool)
    pres = np.zeros(K, bool); cent = np.full((K, 3), np.nan, np.float32); meth = []
    for k, r in enumerate(res):
        if r is None:
            meth.append("absent"); continue
        h, Rk, c, mt = r
        H[k] = np.clip(h, -1024, 3071).astype(np.int16); R[k] = Rk; pres[k] = Rk.any(); cent[k] = c; meth.append(mt)
    np.savez_compressed(fo + ".tmp.npz", present=pres, hu=H, regions=np.packbits(R.ravel()), rshape=np.array(R.shape),
                        center=cent, method=np.array(meth), hu_feats=hu_feats(H, R), orient=np.array(o))
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
