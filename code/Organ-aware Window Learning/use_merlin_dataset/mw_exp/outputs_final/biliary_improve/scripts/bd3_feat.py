"""BD1/BD3 per-case features from 3 mm prep npz (hu z,y,x) + TotalSegmentator 'total' mask (x,y,z -> .T).
Usage: bd3_feat.py ids.txt prep_dir mask_dir out.csv nproc
 (ids = train/val only; nothing is read from test).
Writes per case:
  * the analysis_biliary/bd_hu.py features (identical code, prefix-free) -> used if hu_a6.csv is still missing
  * BD3 intrahepatic dark-tube detector: multi-scale Hessian (Frangi, black ridges) inside the 1-voxel-eroded
    liver, dark = 0..25 HU (absolute) or < liver_median-35 (relative); portal/splenic vein + IVC (dilated 1 vox)
    excluded. tube volume, largest tube, max diameter (EDT) overall and near the hilum (<=30 mm from PV), skeleton
    branch points / endpoints.
  * extended porta-hepatis / hepatoduodenal-ligament CBD proxy at 3 mm: unlabelled voxels within 24 mm of the
    portal vein, between liver and pancreas (inside dilation of both), fluid (-5..30 HU) connected components;
    largest component volume, max EDT diameter, cranio-caudal length, tubularity, and the same for the
    pancreatic-head corridor (distal CBD) incl. a 'tube' running from porta to pancreatic head.
"""
import sys, os, numpy as np, pandas as pd, warnings
from multiprocessing import Pool
from scipy import ndimage
import nibabel as nib
warnings.filterwarnings("ignore")
C = dict(gb=4, liver=5, stomach=6, panc=7, duod=19, aorta=52, ivc=63, pv=64, spleen=1)
VOX = 27.0 / 1000.0  # ml per 3 mm voxel
ball = lambda r: (np.add.outer(np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2), np.arange(-r, r + 1) ** 2) <= r * r)


def dil(mask, r):
    """== ndimage.binary_dilation(mask, structure=ball(r)) (ball = Euclidean distance <= r), via EDT (much faster)."""
    if not mask.any():
        return np.zeros_like(mask, bool)
    return ndimage.distance_transform_edt(~mask) <= r


def med(hu, m):
    return float(np.median(hu[m])) if m.sum() > 20 else np.nan


def ccstats(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return 0, 0
    s = np.bincount(lab.ravel())[1:]
    return int(s.max()), int((s >= 3).sum())


def hu_feats(hu, m, M, d):
    """verbatim analysis_biliary/scripts/bd_hu.py feature block."""
    for k in ("liver", "panc", "gb", "duod", "pv"):
        d[f"{k}_vox"] = int(M[k].sum())
    for k in ("aorta", "pv", "ivc", "liver", "spleen", "panc"):
        d[f"{k}_hu"] = med(hu, M[k])
    L = M["liver"]
    if L.sum() < 500:
        return
    lmed = d["liver_hu"]
    Le = ndimage.binary_erosion(L, iterations=1)
    d["liver_hu_sd"] = float(hu[Le].std()) if Le.sum() > 20 else np.nan
    low = Le & (hu < lmed - 35) & (hu > -20)
    d["liv_low_n"] = int(low.sum()); d["liv_low_frac"] = float(low.sum() / max(Le.sum(), 1))
    d["liv_low_maxcc"], d["liv_low_ncc"] = ccstats(low)
    fl = Le & (hu >= -5) & (hu <= 25)
    d["liv_fluid_n"] = int(fl.sum())
    d["liv_air_n"] = int((L & (hu < -300)).sum())
    pv = M["pv"]
    if pv.sum() >= 20:
        pvd = dil(pv, 7)
        hil = Le & pvd & ~pv
        d["hil_vox"] = int(hil.sum())
        hlow = hil & (hu < lmed - 35) & (hu > -20)
        d["hil_low_n"] = int(hlow.sum()); d["hil_low_frac"] = float(hlow.sum() / max(hil.sum(), 1))
        d["hil_low_maxcc"], _ = ccstats(hlow)
        near = dil(pv, 5) & (m == 0)
        near &= dil(L | M["panc"], 10)
        d["porta_vox"] = int(near.sum())
        pf = near & (hu >= -5) & (hu <= 30)
        d["porta_fluid_n"] = int(pf.sum()); d["porta_fluid_frac"] = float(pf.sum() / max(near.sum(), 1))
        d["porta_fluid_maxcc"], _ = ccstats(pf)
        d["porta_air_n"] = int((near & (hu < -300)).sum())
        d["porta_metal_n"] = int((dil(pv, 8) & (m == 0) & (hu > 600)).sum())
        d["porta_hu_med"] = med(hu, near)
    P, D = M["panc"], M["duod"]
    if P.sum() >= 50:
        Pe = ndimage.binary_erosion(P, iterations=1)
        pmed = d["panc_hu"]
        d["panc_low_n"] = int((Pe & (hu < pmed - 30) & (hu > -20)).sum())
        d["panc_low_frac"] = float(d["panc_low_n"] / max(Pe.sum(), 1))
        if D.sum() >= 50:
            reg = dil(P, 3) & dil(D, 3) & (m == 0)
            d["phead_vox"] = int(reg.sum())
            d["phead_fluid_n"] = int((reg & (hu >= -5) & (hu <= 30)).sum())
            d["phead_metal_n"] = int((reg & (hu > 600)).sum())


def bbox(mask, pad, shape):
    w = np.where(mask)
    return tuple(slice(max(int(a.min()) - pad, 0), min(int(a.max()) + pad + 1, s)) for a, s in zip(w, shape))


def largest_cc(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None, 0
    s = np.bincount(lab.ravel()); s[0] = 0
    return lab == s.argmax(), n


def comp_stats(cc, pre, d):
    """volume, max diameter (2*EDT*3mm), z-length (mm), tubularity of a single component."""
    if cc is None or cc.sum() == 0:
        for k in ("vol", "dmax", "zlen", "tub"):
            d[f"{pre}_{k}"] = 0.0
        return
    edt = ndimage.distance_transform_edt(cc)
    zz = np.where(cc.any(axis=(1, 2)))[0]
    d[f"{pre}_vol"] = float(cc.sum() * VOX)
    d[f"{pre}_dmax"] = float(2 * edt.max() * 3.0)
    d[f"{pre}_zlen"] = float((zz.max() - zz.min() + 1) * 3.0)
    d[f"{pre}_tub"] = float(d[f"{pre}_zlen"] / max(np.cbrt(cc.sum() * 27.0), 1e-3))


def skel_counts(mask):
    from skimage.morphology import skeletonize
    if mask.sum() < 3:
        return 0, 0, 0
    sk = skeletonize(mask).astype(bool) if mask.ndim == 3 else mask
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - 1
    nb = nb * sk
    return int(sk.sum()), int((nb >= 3).sum()), int((nb == 1).sum())


def tube_feats(hu, m, M, d):
    from skimage.filters import frangi
    L = M["liver"]
    if L.sum() < 500:
        return
    lmed = float(np.median(hu[L]))
    sl = bbox(L, 3, hu.shape)
    h = hu[sl].astype(np.float32); Lc = L[sl]; mc = m[sl]
    Le = ndimage.binary_erosion(Lc, iterations=1)
    vess = ndimage.binary_dilation((mc == C["pv"]) | (mc == C["ivc"]), iterations=1)
    x = (np.clip(h, -50, 200) + 50) / 250.0
    V = frangi(x, sigmas=(0.6, 1.0, 1.5, 2.0), black_ridges=True, alpha=0.5, beta=0.5, gamma=None)
    V = np.nan_to_num(V) * Le * ~vess
    pv = (mc == C["pv"])
    hil = ndimage.distance_transform_edt(~pv) <= 10 if pv.sum() >= 20 else np.zeros_like(Lc)  # <=30 mm from PV
    d["tube_vmean_liver"] = float(V[Le].mean()) if Le.sum() else np.nan
    nle = max(Le.sum(), 1)
    for tag, dark in (("abs", (h >= 0) & (h <= 25)), ("rel", (h < lmed - 35) & (h > -20))):
        dk = dark & Le & ~vess
        d[f"tube_{tag}_dark_ml"] = float(dk.sum() * VOX)
        for thr in (0.01, 0.03, 0.08):
            t = dk & (V > thr)
            key = f"tube_{tag}_t{int(round(thr * 100)):02d}"
            d[f"{key}_ml"] = float(t.sum() * VOX)
            d[f"{key}_frac"] = float(t.sum() / nle)
            cc, n = largest_cc(t)
            d[f"{key}_ncc"] = int(n)
            d[f"{key}_maxcc_ml"] = float(cc.sum() * VOX) if cc is not None else 0.0
            edt = ndimage.distance_transform_edt(t) if t.any() else np.zeros(t.shape)
            d[f"{key}_dmax"] = float(2 * edt.max() * 3.0)
            d[f"{key}_dmax_hil"] = float(2 * (edt * hil).max() * 3.0) if hil.any() else 0.0
            d[f"{key}_hil_ml"] = float((t & hil).sum() * VOX)
            if thr == 0.03:
                ns, nbr, nend = skel_counts(t)
                d[f"{key}_skel"] = ns; d[f"{key}_branch"] = nbr; d[f"{key}_ends"] = nend


def porta_feats(hu, m, M, d):
    pv, L, P, D = M["pv"], M["liver"], M["panc"], M["duod"]
    if pv.sum() < 20 or L.sum() < 500:
        return
    sl = bbox(pv | P, 10, hu.shape)
    h = hu[sl]; mc = m[sl]; pvc = pv[sl]; Lc = L[sl]; Pc = P[sl]; Dc = D[sl]
    dpv = ndimage.distance_transform_edt(~pvc)
    dl = ndimage.distance_transform_edt(~Lc); dp = ndimage.distance_transform_edt(~Pc) if Pc.any() else np.full(h.shape, 99.)
    free = (mc == 0)
    fluid = (h >= -5) & (h <= 30)
    # hepatoduodenal ligament corridor: <=24 mm from PV, <=30 mm from liver or pancreas, unlabelled
    hdl = free & (dpv <= 8) & ((dl <= 10) | (dp <= 10))
    d["hdl_vox"] = int(hdl.sum())
    cc, n = largest_cc(hdl & fluid)
    comp_stats(cc, "hdl_fl", d); d["hdl_fl_ncc"] = int(n)
    d["hdl_fl_ml"] = float((hdl & fluid).sum() * VOX)
    # stricter water-like (-5..20)
    cc2, _ = largest_cc(hdl & (h >= -5) & (h <= 20))
    comp_stats(cc2, "hdl_w", d)
    # fluid component touching both liver hilum zone and pancreas zone (continuous duct porta->head)
    if Pc.any():
        lab, n = ndimage.label(free & fluid & ((dpv <= 8) | (dp <= 4)), structure=np.ones((3, 3, 3)))
        best = None
        if n:
            touchL = set(np.unique(lab[(dl <= 3) & (lab > 0)])); touchP = set(np.unique(lab[(dp <= 2) & (lab > 0)]))
            both = list((touchL & touchP) - {0})
            if both:
                sz = ndimage.sum(np.ones_like(lab), lab, both); best = lab == both[int(np.argmax(sz))]
        comp_stats(best, "duct_lp", d)
        # pancreatic head / distal CBD corridor
        if Dc.sum() >= 50:
            dd = ndimage.distance_transform_edt(~Dc)
            ph = free & (dp <= 3) & (dd <= 5)
            ph |= Pc & (h < (np.median(h[Pc]) - 30)) & (h > -20) & (dd <= 7)
            cc3, _ = largest_cc(ph & fluid | (ph & Pc))
            comp_stats(cc3, "phd_fl", d)


def feat(args):
    sid, prep, mdir = args
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
    if os.environ.get("NOHU"):
        for k in ("liver", "panc", "gb", "duod", "pv"):
            d[f"{k}_vox"] = int(M[k].sum())
    fns = (tube_feats, porta_feats) if os.environ.get("NOHU") else (hu_feats, tube_feats, porta_feats)
    for fn in fns:
        try:
            fn(hu, m, M, d)
        except Exception as e:  # noqa: BLE001
            d[f"err_{fn.__name__}"] = str(e)[:80]
    return d


if __name__ == "__main__":
    ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    L_ = [l.strip() for l in open(ids) if l.strip()]
    import json
    part = out + ".part.jsonl"
    done = set()
    if os.path.exists(part):
        for l in open(part):
            try: done.add(json.loads(l)["study_id"])
            except Exception: pass
    todo = [(s, prep, mdir) for s in L_ if s not in done]
    with Pool(npr) as p, open(part, "a") as fh:
        for i, r in enumerate(p.imap_unordered(feat, todo, chunksize=4)):
            if r is not None:
                fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 200 == 0:
                print("progress", i + 1, len(todo), flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(part)]).drop_duplicates("study_id")
    df.to_csv(out, index=False)
    print("done", len(df), "of", len(L_), flush=True)
