"""Task B: GB-fossa high-density search for GB-missing/tiny cases (3 mm TS GB < 100 vox). Train/val ids only; no labels read.
Inputs: 3 mm prep npz (local dir, else fetched from CT-rate and deleted after use), TS total 3 mm mask, GS6 1.5 mm GB mask.
Anchors: P_exp = liver-bbox fraction point [0.3355, 0.7002, 0.6623] (Exp N/GS5 fossa); P_gs6 = GS6 GB centroid (3 mm grid,
linear >= 0.25) if >= 3 vox; P_pv = portal-vein voxels within 10 mm of the liver (porta) centroid.
A = P_gs6 if available else P_exp.  Fossa region R_r (r = 30, 50 mm around A): liver depth <= 6 mm, within 15 mm outside the
liver, minus bone(+1 vox)/vessels/kidney/spleen/pancreas/stomach/colon/small bowel/duodenum.
Per R: >130 HU components (26-conn): counts (all / 130-1000 / metal >= 1500), volume, max (non-metal), largest blob vol + mean HU,
shell index of the largest non-metal blob (per-axial-slice hole filling: filled/vol), min dist of blobs to liver surface,
dist of top blob to A and to P_pv; fluid pocket (-10..45 HU, largest component vol + mean HU); duodenum-labelled >130 HU voxels
within R50 (oral contrast confound).  GS6 GB (3 mm, dilated 1 vox): max HU, n > 130, vox.
usage: gbf_fossa.py ids.txt out.csv nproc prepdir maskmap.tsv gs6dir tmpdir"""
import sys, os, subprocess
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
from totalsegmentator.map_to_binary import class_map
CM = class_map["total"]; INV = {v: k for k, v in CM.items()}
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
OTH = ids_of("kidney", "spleen", "pancreas", "stomach", "small_bowel", "colon", "esophagus", "adrenal")
LIV, DUO, PV, GBL = INV["liver"], INV["duodenum"], INV["portal_vein_and_splenic_vein"], INV["gallbladder"]
EXP = np.array([0.3355, 0.7002, 0.6623]); SP = 3.0; ST26 = np.ones((3, 3, 3), bool)
IDS, OUT, NP, PREP, MM, GS6, TMP = sys.argv[1:8]; NP = int(NP)
MMAP = dict(l.rstrip("\n").split("\t") for l in open(MM))
CTR = "daikso@192.168.0.189:OWL/mw_exp/outputs/prep"


def get_prep(sid):
    f = f"{PREP}/{sid}.npz"
    if os.path.exists(f): return f, False
    t = f"{TMP}/{sid}.npz"
    r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{CTR}/{sid}.npz", t], capture_output=True)
    return (t, True) if r.returncode == 0 and os.path.exists(t) else (None, False)


def region_feats(hu, m, R, A, Ppv, Lsurf_out, Ld, tag, thr=130.0):
    f = {}
    f[f"{tag}_vox"] = int(R.sum())
    if R.sum() < 10: return f
    cand = R & (hu > thr); lab, n = ndimage.label(cand, ST26)
    f[f"{tag}_n130"] = int(cand.sum())
    if n:
        idx = np.arange(1, n + 1)
        sz = ndimage.sum(np.ones_like(hu), lab, idx); mx = ndimage.maximum(hu, lab, idx); mn = ndimage.mean(hu, lab, idx)
        metal = mx >= 1500; nm = ~metal
        f[f"{tag}_nblob"] = int(n); f[f"{tag}_nblob_metal"] = int(metal.sum()); f[f"{tag}_metal_max"] = float(mx[metal].max()) if metal.any() else 0.0
        f[f"{tag}_nblob_st"] = int(nm.sum()); f[f"{tag}_nblob_st2"] = int((nm & (sz >= 2)).sum())
        f[f"{tag}_nblob_200"] = int((nm & (mx >= 200)).sum()); f[f"{tag}_nblob_400"] = int((nm & (mx >= 400)).sum())
        if nm.any():
            f[f"{tag}_vol_st"] = float(sz[nm].sum() * SP ** 3); f[f"{tag}_max_st"] = float(mx[nm].max())
            b = idx[nm][np.argmax(sz[nm] * (mx[nm] >= 160) + 1e-3 * mx[nm])]
            f[f"{tag}_big_vol"] = float(sz[b - 1] * SP ** 3); f[f"{tag}_big_mean"] = float(mn[b - 1]); f[f"{tag}_big_max"] = float(mx[b - 1])
            bm = lab == b; sl = ndimage.find_objects(bm.astype(np.int8))[0]; sub = bm[sl]
            filled = np.stack([ndimage.binary_fill_holes(s) for s in sub]).sum()
            f[f"{tag}_big_shell"] = float(filled / max(sub.sum(), 1))
            # top (max HU non-metal) blob position
            t = idx[nm][np.argmax(mx[nm])]; pos = np.array(ndimage.maximum_position(hu, lab, t), float)
            f[f"{tag}_top_dA"] = float(np.linalg.norm(pos - A) * SP)
            f[f"{tag}_top_dpv"] = float(np.linalg.norm(pos - Ppv) * SP) if Ppv is not None else np.nan
            stv = np.isin(lab, idx[nm])
            f[f"{tag}_min_dliv"] = float(np.minimum(Lsurf_out[stv], 99).min())
            f[f"{tag}_frac_inliver"] = float((Ld[stv] > 0).mean())
            # sum of blob "mass" (HU-130)*vol for non-metal
            f[f"{tag}_mass"] = float(np.clip(hu[stv] - thr, 0, 1500).sum() * SP ** 3 / 1000)
    fl = R & (hu >= -10) & (hu <= 45)
    if fl.any():
        lf, nf = ndimage.label(fl, ST26); s = ndimage.sum(np.ones_like(hu), lf, np.arange(1, nf + 1)); k = int(np.argmax(s)) + 1
        f[f"{tag}_fluid_big"] = float(s[k - 1] * SP ** 3); f[f"{tag}_fluid_mean"] = float(hu[lf == k].mean())
        f[f"{tag}_fluid_tot"] = float(fl.sum() * SP ** 3)
    f[f"{tag}_fat_frac"] = float((hu[R] < -30).mean())
    return f


def run(sid):
    try:
        fp, tmp = get_prep(sid)
        if fp is None: return dict(study_id=sid, err="no_prep")
        z = np.load(fp); hu = z["hu"].astype(np.float32); aff = z["affine"]; shp = z["units"].shape
        names = list(z["unit_names"]); gb3 = z["units"] == names.index("gallbladder") + 1
        if tmp: os.remove(fp)
        m = np.asanyarray(nib.load(MMAP[sid]).dataobj).astype(np.int16).transpose(2, 1, 0)
        f = dict(study_id=sid, gb3_vox=int(gb3.sum()))
        L = m == LIV
        if L.sum() < 1000: f["err"] = "no_liver"; return f
        lz, ly, lx = [np.where(L.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
        lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float); Pexp = lo + EXP * (hi - lo)
        # GS6
        g6f = f"{GS6}/{sid}.nii.gz"; g6 = None
        if os.path.exists(g6f):
            from nibabel.processing import resample_from_to
            seg = nib.load(g6f); g = (np.asanyarray(seg.dataobj) == 1)
            f["gs6_vox15"] = int(g.sum())
            if g.any():
                r = resample_from_to(nib.Nifti1Image(g.astype(np.float32), seg.affine), (shp[::-1], aff), order=1)
                g6 = (np.asanyarray(r.dataobj) >= .25).transpose(2, 1, 0)
        f["gs6_vox3"] = int(g6.sum()) if g6 is not None else 0
        A = np.array(ndimage.center_of_mass(g6)) if g6 is not None and g6.sum() >= 3 else Pexp
        f["anchor_gs6"] = int(g6 is not None and g6.sum() >= 3)
        f["d_gs6_exp"] = float(np.linalg.norm(A - Pexp) * SP)
        # crop +-25 vox (75 mm) around the anchor before distance transforms
        lo3 = np.maximum(np.round(A).astype(int) - 25, 0); hi3 = np.round(A).astype(int) + 26
        sl = tuple(slice(a_, b_) for a_, b_ in zip(lo3, hi3)); off = lo3.astype(float)
        hu, m, L = hu[sl], m[sl], L[sl]; g6 = g6[sl] if g6 is not None else None; A = A - off; Pexp = Pexp - off; shp = hu.shape
        Lout = ndimage.distance_transform_edt(~L) * SP if L.any() else np.full(shp, 99.0); Ld = ndimage.distance_transform_edt(L) * SP
        pv = (m == PV) & (Lout <= 10); Ppv = np.array(ndimage.center_of_mass(pv)) if pv.sum() >= 5 else None
        f["d_exp_pv"] = float(np.linalg.norm(Pexp - Ppv) * SP) if Ppv is not None else np.nan
        f["d_A_liver"] = float(Lout[tuple(np.clip(np.round(A).astype(int), 0, np.array(shp) - 1))])
        if g6 is not None and g6.any():
            gd = ndimage.binary_dilation(g6, iterations=1); v = hu[gd]
            f.update(gs6_max=float(v.max()), gs6_n130=int((v > 130).sum()), gs6_n300=int((v > 300).sum()), gs6_med=float(np.median(hu[g6])))
        excl = ndimage.binary_dilation(np.isin(m, BONE), iterations=1) | np.isin(m, VESS + OTH + [DUO])
        zz, yy, xx = np.ogrid[:shp[0], :shp[1], :shp[2]]
        dA = np.sqrt((zz - A[0]) ** 2 + (yy - A[1]) ** 2 + (xx - A[2]) ** 2) * SP
        lmed = float(np.median(hu[L & (Ld > 9)])) if (L & (Ld > 9)).sum() > 100 else float(np.median(hu[L]))
        thr = max(130.0, lmed + 50.0); f["liver_med"] = lmed; f["blob_thr"] = thr
        base = (Ld <= 3) & (Lout <= 15) & ~excl
        for r in (30, 50):
            f.update(region_feats(hu, m, base & (dA <= r), A, Ppv, Lout, Ld, f"fs{r}", thr))
        R50 = (Ld <= 6) & (Lout <= 15) & (dA <= 50)
        f["fs50_duo_n130"] = int((R50 & (m == DUO) & (hu > 130)).sum())
        f["fs50_colon_n130"] = int((R50 & np.isin(m, ids_of("colon")) & (hu > 130)).sum())
        f["body_metal_n"] = int((hu >= 2000).sum())
        return f
    except Exception as e:
        return dict(study_id=sid, err=str(e)[:200])


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    L = [l.strip() for l in open(IDS) if l.strip()]
    done = set()
    if os.path.exists(OUT): done = set(pd.read_csv(OUT).study_id)
    L = [s for s in L if s not in done]
    R = []
    with Pool(NP) as p:
        for i, r in enumerate(p.imap_unordered(run, L, chunksize=4)):
            R.append(r)
            if (i + 1) % 200 == 0:
                print(i + 1, len(L), flush=True)
                pd.DataFrame(R).to_csv(OUT + ".part", index=False)
    old = [pd.read_csv(OUT)] if os.path.exists(OUT) else []
    pd.concat(old + [pd.DataFrame(R)]).to_csv(OUT, index=False); print("DONE", len(R), flush=True)
