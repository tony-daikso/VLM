"""gs15 stage A (CPU worker; A6000 or Mac): 1.5 mm gallbladder / GB-fossa crop from the ORIGINAL CT (train/val ids only).
Centre (same rule as Exp N nu_crop.py 'gs', world coordinates):
  TS 3 mm 'total' mask gallbladder (label 4) >= 100 vox      -> GB centroid                          method 'gb'
  else GS6 1.5 mm GB re-segmentation (label 1) >= 800 vox    -> GS6 GB centroid                      method 'gs6_gb'
  else liver (label 5) >= 1000 vox                           -> expected GB-fossa point (liver bbox fraction
                                                                [0.3355, 0.7002, 0.6623] in the z,y,x mask array, = GS5/Exp N)
                                                                method 'fossa' / 'gs6_fossa'
  else any GB voxel -> 'gbtiny'; else absent.
GB mask on the crop grid: GS6 mask when the case has one (GS6 cases = 3 mm GB < 100 vox), else TS 3 mm GB (linear, >= 0.5).
Original CT -> canonical RAS -> sub-box -> anti-alias Gaussian (axes with spacing < 1.5 mm) -> trilinear onto a 96^3 x 1.5 mm
axis-aligned RAS grid (144 mm cube), pad -1024; 3 mm labels nearest-neighbour. Arrays stored (z,y,x) (+y = anterior).
Regions (bool):
  r0 gb_dil   : GB dilated 3 mm (EDT <= 3 mm)
  r1 fossa30  : 30 mm ball around the expected fossa point minus bone(+6 mm) / vessels / uro / bowel(+3 mm) / liver core (> 6 mm
                inside the liver)  (= GS5/Exp N fossa30 at 1.5 mm)
  r2 dep_third: dependent (posterior, lowest-y) third of the GB  (supine: dependent layer)
  r3 ball20   : 20 mm ball around the crop centre (always present)
  r4 dense    : r0 & HU >= 90 (hyperdense stone candidates; may be empty)
HU features (1.5 mm, written to logs/crop_info_<tag>.jsonl 'f' dict and the npz 'feat'): GB volume, percentiles, max, std,
  counts >= 60/100/130/200/300 HU (in mL), low-density counts (<= -10 HU; < GB median - 25 HU after 3 mm smoothing = cholesterol
  stone candidates), largest blob >= 100 HU, n blobs, dependent-layer median minus non-dependent median, dependent p95, wall shell
  p99/n>=130, fossa ROI counts >= 130/300 HU, max, p99, liver median, GB-to-liver median difference.
Claims (O_EXCL files in claims/) make workers disjoint; STOP stops after the current case.
Usage: gs15_crop.py --ids ids.txt --maskmap maskmap.tsv --gs6 GS6DIR --root ROOT --wid 0 [--ct-dir LOCALDIR | --ct-ssh]
                    [--max-cases 300] [--threads 2] [--tag a6]"""
import argparse, os, sys, time, json, subprocess
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True); ap.add_argument("--maskmap", required=True); ap.add_argument("--root", required=True)
ap.add_argument("--gs6", required=True); ap.add_argument("--ct-dir", default=None); ap.add_argument("--ct-ssh", action="store_true")
ap.add_argument("--wid", type=int, default=0); ap.add_argument("--max-cases", type=int, default=300)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tag", default="a6"); ap.add_argument("--one", default=None)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = str(a.threads)
import nibabel as nib
from scipy import ndimage

CT_HOST = "daikso@192.168.0.189"
CT_DIR_SSH = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
GB, LIV = 4, 5
BONE_IDS = list(range(25, 51)) + list(range(69, 79)) + list(range(91, 118))
VESS = list(range(51, 69)); BOWEL = [6, 15, 18, 19, 20]; URO = [2, 3, 21, 23, 24]
LUT_B = np.zeros(256, bool); LUT_B[BONE_IDS] = True
LUT_V = np.zeros(256, bool); LUT_V[VESS + URO] = True
LUT_W = np.zeros(256, bool); LUT_W[BOWEL] = True
EXP = np.array([0.3355, 0.7002, 0.6623])
S, SP = 96, 1.5
R = Path(a.root).expanduser()
for d in ("crops", "claims", "ct", "logs"):
    (R / d).mkdir(parents=True, exist_ok=True)
mm = dict(l.rstrip("\n").split("\t") for l in open(a.maskmap) if "\t" in l)


def claim(cid):
    try:
        os.close(os.open(R / "claims" / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def fetch_ct(cid):
    if a.ct_dir:
        return Path(a.ct_dir) / f"{cid}.nii.gz", False
    dst = R / "ct" / f"{cid}.nii.gz"; tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["ssh", "-c", "aes128-gcm@openssh.com", "-o", "BatchMode=yes", CT_HOST, f"cat {CT_DIR_SSH}/{cid}.nii.gz"],
                       stdout=fh, check=True, timeout=900)
    tmp.rename(dst); return dst, True


def world(aff, ijk):
    return (aff @ np.r_[np.asarray(ijk, float), 1])[:3]


def centre(cid, m3, lab):
    """returns centre world, fossa world (or None), method, gs6 image (or None)"""
    gbv = int((lab == GB).sum()); g6 = None; tag = ""
    f6 = Path(a.gs6) / f"{cid}.nii.gz"
    fossa = None
    L = lab == LIV
    if L.sum() >= 1000:
        Lt = L.T                                                    # (z,y,x) like the Exp N prep arrays
        lz, ly, lx = [np.where(Lt.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1))]
        lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float)
        pt = lo + EXP * (hi - lo)
        fossa = world(m3.affine, pt[::-1])
    if gbv >= 100:
        return world(m3.affine, np.argwhere(lab == GB).mean(0)), fossa, "gb", None
    if f6.exists():
        g6 = nib.load(str(f6)); tag = "gs6_"
        G6 = np.asarray(g6.dataobj) == 1
        if G6.sum() >= 800:
            return world(g6.affine, np.argwhere(G6).mean(0)), fossa, "gs6_gb", g6
    if fossa is not None:
        return fossa, fossa, tag + "fossa", g6
    if gbv > 0:
        return world(m3.affine, np.argwhere(lab == GB).mean(0)), None, tag + "gbtiny", g6
    return None, None, "absent", g6


def grid_world(cw):
    off = (np.arange(S) - (S - 1) / 2) * SP
    X, Y, Z = np.meshgrid(cw[0] + off, cw[1] + off, cw[2] + off, indexing="ij")
    return np.stack([X, Y, Z, np.ones_like(X)], 0).reshape(4, -1)


def samp(arr, aff, G, order):
    v = np.linalg.inv(aff) @ G
    return ndimage.map_coordinates(arr, v[:3], order=order, mode="constant", cval=0).reshape(S, S, S)


def feats(hu, gb, r1, dep, lab):
    f = {}
    v = hu[gb].astype(np.float32); vml = 1.5 ** 3 / 1000.0
    f["gb_ml"] = float(gb.sum() * vml)
    Lc = lab == LIV
    lm = float(np.median(hu[Lc])) if Lc.sum() > 200 else np.nan
    f["liver_med"] = lm
    if v.size >= 30:
        for p in (5, 25, 50, 75, 90, 95, 99):
            f[f"gb_p{p}"] = float(np.percentile(v, p))
        f["gb_max"] = float(v.max()); f["gb_std"] = float(v.std()); f["gb_min"] = float(v.min())
        for t in (60, 100, 130, 200, 300):
            f[f"gb_ml{t}"] = float((v >= t).sum() * vml)
        f["gb_frac100"] = float((v >= 100).mean())
        f["gb_ml_low"] = float((v <= -10).sum() * vml)
        sm = ndimage.uniform_filter(hu.astype(np.float32), 3)
        med = float(np.median(v)); g_in = ndimage.binary_erosion(gb, iterations=1)
        lowb = g_in & (sm < med - 25)
        f["gb_ml_chol"] = float(lowb.sum() * vml)
        lab_l, nl = ndimage.label(lowb); f["gb_chol_maxblob"] = float(np.bincount(lab_l.ravel())[1:].max() * vml) if nl else 0.0
        dense = gb & (hu >= 100)
        lab_d, nd = ndimage.label(dense)
        f["gb_nblob100"] = float(nd); f["gb_maxblob100"] = float(np.bincount(lab_d.ravel())[1:].max() * vml) if nd else 0.0
        hi_ = gb & (sm >= med + 25); f["gb_ml_hi25"] = float(hi_.sum() * vml)
        if dep.sum() >= 10 and (gb & ~dep).sum() >= 10:
            f["dep_minus_nondep"] = float(np.median(hu[dep]) - np.median(hu[gb & ~dep])); f["dep_p95"] = float(np.percentile(hu[dep], 95))
        f["gb_minus_liver"] = med - lm if not np.isnan(lm) else np.nan
        shell = (ndimage.distance_transform_edt(~gb, sampling=SP) <= 4.5) & ~gb
        sv = hu[shell].astype(np.float32)
        if sv.size >= 30:
            f["shell_p99"] = float(np.percentile(sv, 99)); f["shell_ml130"] = float((sv >= 130).sum() * vml)
    w = hu[r1].astype(np.float32)
    if w.size >= 30:
        f["fos_ml130"] = float((w >= 130).sum() * vml); f["fos_ml300"] = float((w >= 300).sum() * vml)
        f["fos_max"] = float(w.max()); f["fos_p99"] = float(np.percentile(w, 99)); f["fos_p50"] = float(np.median(w))
        f["fos_ml_fluid"] = float(((w >= -10) & (w <= 30)).sum() * vml)
    return f


def one(cid):
    t0 = time.time()
    mp = mm.get(cid)
    if mp is None or not os.path.exists(mp):
        return dict(status="nomask")
    m3 = nib.load(mp); lab3 = np.asarray(m3.dataobj).astype(np.uint8)
    cw, fw, meth, g6 = centre(cid, m3, lab3)
    if cw is None:
        np.savez(R / "crops" / f".{cid}.tmp.npz", present=np.array([False]), rshape=np.array([1, 5, S, S, S]), method=np.array([meth]))
        os.replace(R / "crops" / f".{cid}.tmp.npz", R / "crops" / f"{cid}.npz")
        return dict(status="absent", method=meth)
    t1 = time.time()
    fc, tmpct = fetch_ct(cid); t2 = time.time()
    try:
        img = nib.as_closest_canonical(nib.load(str(fc)))
        zooms = np.array(img.header.get_zooms()[:3], float)
        G = grid_world(cw)
        vox = np.linalg.inv(img.affine) @ G
        lo = np.floor(vox[:3].min(1)).astype(int) - 4; hi = np.ceil(vox[:3].max(1)).astype(int) + 5
        lo = np.maximum(lo, 0); hi = np.minimum(hi, img.shape[:3])
        if np.any(hi - lo < 2):
            return dict(status="outside", method=meth)
        sub = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    finally:
        if tmpct:
            fc.unlink(missing_ok=True)
    sig = np.array([0.5 * np.sqrt(max((SP / z) ** 2 - 1, 0)) if z < SP else 0 for z in zooms])
    if sig.max() > 0:
        sub = ndimage.gaussian_filter(sub, sig, mode="nearest")
    hu = ndimage.map_coordinates(sub, vox[:3] - lo[:, None], order=1, mode="constant", cval=-1024).reshape(S, S, S)
    del sub
    hu = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)
    lab = samp(lab3, m3.affine, G, 0).astype(np.uint8)
    if g6 is not None:
        gbm = samp((np.asarray(g6.dataobj) == 1).astype(np.float32), g6.affine, G, 1) >= 0.5
    else:
        gbm = samp((lab3 == GB).astype(np.float32), m3.affine, G, 1) >= 0.5
    t3 = time.time()
    hu = hu.transpose(2, 1, 0).copy(); lab = lab.transpose(2, 1, 0).copy(); gbm = gbm.transpose(2, 1, 0).copy()
    zz, yy, xx = np.indices((S, S, S)); c0 = (S - 1) / 2
    r0 = (ndimage.distance_transform_edt(~gbm, sampling=SP) <= 3.0) if gbm.any() else np.zeros_like(gbm)
    r1 = np.zeros_like(gbm)
    if fw is not None:
        p = (fw - cw) / SP + c0                                     # (x,y,z) grid index
        p = p[::-1]                                                 # -> (z,y,x)
        ball = ((zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2) * SP ** 2 <= 30 ** 2
        B, V, W, Lc = LUT_B[lab], LUT_V[lab], LUT_W[lab], lab == LIV
        excl = V
        if B.any(): excl = excl | (ndimage.distance_transform_edt(~B, sampling=SP) <= 6)
        if W.any(): excl = excl | (ndimage.distance_transform_edt(~W, sampling=SP) <= 3)
        core = ndimage.distance_transform_edt(Lc, sampling=SP) > 6 if Lc.any() else np.zeros_like(Lc)
        r1 = ball & ~excl & ~core
    dep = np.zeros_like(gbm)
    if gbm.sum() >= 3:
        yv = np.argwhere(gbm)[:, 1]; dep = gbm & (np.arange(S)[None, :, None] <= np.percentile(yv, 33.3))
    r3 = ((zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2) * SP ** 2 <= 20 ** 2
    r4 = r0 & (hu >= 90)
    Rg = np.stack([r0, r1, dep, r3, r4])[None]
    f = feats(hu, gbm, r1, dep, lab)
    fk = sorted(f)
    tmp = R / "crops" / f".{cid}.tmp.npz"
    np.savez_compressed(tmp, present=np.array([True]), hu=hu[None], regions=np.packbits(Rg.ravel()), rshape=np.array(Rg.shape),
                        gbmask=np.packbits(gbm.ravel()), center_world=cw, fossa_world=(fw if fw is not None else np.full(3, np.nan)),
                        ct_zooms=zooms, method=np.array([meth]), feat=np.array([f[k] for k in fk], np.float32), feat_names=np.array(fk))
    os.replace(tmp, R / "crops" / f"{cid}.npz")
    return dict(status="ok", method=meth, vox=[int(x) for x in Rg[0].sum((1, 2, 3))], zooms=zooms.round(3).tolist(), f=f,
                t_mask=round(t1 - t0, 2), t_ct=round(t2 - t1, 2), t_resample=round(t3 - t2, 2), t_regions=round(time.time() - t3, 2))


if a.one:
    r = one(a.one); print(json.dumps(r, default=float)); sys.exit(0)
ids = [l.strip() for l in open(a.ids) if l.strip()]
info = open(R / "logs" / f"crop_info_{a.tag}_w{a.wid}.jsonl", "a"); n = 0
for cid in ids:
    if (R / "STOP").exists() or (R / f"STOP_{a.tag}").exists() or n >= a.max_cases:
        break
    if (R / "crops" / f"{cid}.npz").exists() or not claim(cid):
        continue
    t = time.time()
    try:
        r = one(cid)
    except Exception as e:  # noqa: BLE001
        r = dict(status="err", err=repr(e)[:200])
        for f_ in (R / "ct").glob(f"{cid}*"):
            f_.unlink(missing_ok=True)
        with open(R / f"failed_{a.tag}.txt", "a") as fh:
            fh.write(f"{cid}\t{r['err']}\n")
    r.update(study_id=cid, wid=a.wid, host=a.tag, t=round(time.time() - t, 2), ts=time.strftime("%H:%M:%S"))
    info.write(json.dumps(r, default=float) + "\n"); info.flush(); n += 1
else:
    if not (R / "STOP").exists() and not (R / f"STOP_{a.tag}").exists():
        (R / f"CROP_DONE_{a.tag}_w{a.wid}").touch()
print("CROP_WORKER_EXIT", a.tag, a.wid, n, flush=True)
