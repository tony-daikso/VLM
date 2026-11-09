"""spine_ft / pelvis_ft stage A (CPU, CT-rate): 1.5 mm bone crops from the raw Merlin CT, one HDD read per case.
Centres come from the existing TS Dataset297 3 mm 'total' masks (canonical 3 mm grid, nii affine). The crop grid is
axis-aligned to world RAS (array order z=S, y=A, x=R axis index), 1.5 mm, sampled from the raw CT with trilinear
interpolation through the raw affine (no full-volume resampling). Masks are sampled nearest-neighbour from the 3 mm map.
Units:
  spine  [10, 40, 48, 40]: T8..L5, centre = 3 mm vertebra-label centroid; mask = that vertebra.
  pelvis [7, 48, 48, 48]: acet_r, acet_l (hip voxels within 15 mm of the same-side femur + femur voxels within 15 mm of
         the hip), pubic (anterior-most 30 % of both hips in their lowest 30 %), sac_r, sac_l (upper 60 % of each sacrum
         half), fem_r, fem_l (top 70 mm of each femur); mask = the unit's bone(s).
Out: crops/{id}.npz  sp_hu int16, sp_m packbits, sp_present, sp_nvox, pv_hu, pv_m, pv_present, pv_nvox (+ centres).
Usage: frac15_crop.py ids.txt out_dir n_proc mask_dir [mask_dir ...]   (ids are train/val only; done ids skipped)"""
import sys, os, time, traceback, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool

RAW = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
SP_LV = [36, 35, 34, 33, 32, 31, 30, 29, 28, 27]  # T8..L5
SAC, FEML, FEMR, HIPL, HIPR = 25, 75, 76, 77, 78
SP_SH, PV_SH, SP = (40, 48, 40), (48, 48, 48), 1.5
STOP = None


def grid(shape):
    z, y, x = [(np.arange(n) - (n - 1) / 2) * SP for n in shape]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 0)  # world offsets (x, y, z) in mm


G_SP, G_PV = grid(SP_SH), grid(PV_SH)


def sample(arr, aff, centre, G, shape, order, cval):
    """sample arr (index i,j,k) at world points centre + G via inverse affine; local sub-block for memory."""
    P = np.linalg.inv(aff) @ np.vstack([G + centre[:, None], np.ones(G.shape[1])])
    ijk = P[:3]
    lo = np.floor(ijk.min(1)).astype(int) - 1; hi = np.ceil(ijk.max(1)).astype(int) + 2
    lo_c = np.maximum(lo, 0); hi_c = np.minimum(hi, arr.shape)
    if np.any(hi_c <= lo_c):
        return np.full(shape, cval, np.float32)
    sub = np.asarray(arr[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]], np.float32)
    out = ndimage.map_coordinates(sub, ijk - lo_c[:, None], order=order, mode="constant", cval=cval, prefilter=False)
    return out.reshape(shape)


def world(aff, idx):  # idx [N,3] voxel -> world [N,3]
    return idx @ aff[:3, :3].T + aff[:3, 3]


def centres(lab, aff):
    sp = []
    for l in SP_LV:
        v = np.argwhere(lab == l)
        sp.append((world(aff, v.mean(0)[None])[0], len(v)) if len(v) >= 30 else (None, len(v)))
    W = {k: (lambda v: world(aff, v.astype(float)) if len(v) else np.zeros((0, 3)))(np.argwhere(lab == k))
         for k in (SAC, FEML, FEMR, HIPL, HIPR)}
    pv = []
    for hip, fem in ((HIPR, FEMR), (HIPL, FEML)):
        h, f = W[hip], W[fem]
        if len(h) >= 30 and len(f) >= 30:
            bb = np.argwhere((lab == hip) | (lab == fem)); lo = np.maximum(bb.min(0) - 7, 0); hi = bb.max(0) + 8
            sub = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            d = ndimage.distance_transform_edt(sub != fem) * 3.0
            near_h = h[d[tuple((np.argwhere(lab == hip) - lo).T)] <= 15]
            d2 = ndimage.distance_transform_edt(sub != hip) * 3.0
            near_f = f[d2[tuple((np.argwhere(lab == fem) - lo).T)] <= 15]
            pts = np.vstack([near_h, near_f])
            pv.append((pts.mean(0), len(pts)) if len(pts) >= 20 else (h[h[:, 2] <= np.quantile(h[:, 2], .3)].mean(0), 0))
        elif len(h) >= 30:
            pv.append((h[h[:, 2] <= np.quantile(h[:, 2], .3)].mean(0), 0))
        else:
            pv.append((None, 0))
    hh = np.vstack([W[HIPR], W[HIPL]])
    if len(hh) >= 60:
        low = hh[hh[:, 2] <= np.quantile(hh[:, 2], .3)]
        ant = low[low[:, 1] >= np.quantile(low[:, 1], .7)]
        pv.append((ant.mean(0), len(ant)))
    else:
        pv.append((None, 0))
    s = W[SAC]
    if len(s) >= 60:
        xm = np.median(s[:, 0])
        for side in (s[s[:, 0] >= xm], s[s[:, 0] < xm]):  # +x = R (RAS world)
            up = side[side[:, 2] >= np.quantile(side[:, 2], .4)]
            pv.append((up.mean(0), len(up)))
    else:
        pv += [(None, 0), (None, 0)]
    for fem in (FEMR, FEML):
        f = W[fem]
        if len(f) >= 30:
            top = f[f[:, 2] >= f[:, 2].max() - 70]
            pv.append((top.mean(0), len(top)))
        else:
            pv.append((None, 0))
    return sp, pv


PV_MASK = [(HIPR, FEMR), (HIPL, FEML), (HIPR, HIPL), (SAC,), (SAC,), (FEMR,), (FEML,)]


def one(args):
    cid, out, mdirs = args
    dst = f"{out}/crops/{cid}.npz"
    if os.path.exists(dst):
        return cid, "skip", 0
    t0 = time.time()
    try:
        mp = next((f"{d}/{cid}.nii.gz" for d in mdirs if os.path.exists(f"{d}/{cid}.nii.gz")), None)
        if mp is None:
            return cid, "nomask", 0
        m = nib.load(mp); lab = np.asarray(m.dataobj).astype(np.uint8); maff = m.affine
        sp, pv = centres(lab, maff)
        raw = nib.load(f"{RAW}/{cid}.nii.gz"); hu = np.asanyarray(raw.dataobj); raff = raw.affine
        res = {}
        for tag, cs, G, sh, masks in (("sp", sp, G_SP, SP_SH, [(l,) for l in SP_LV]), ("pv", pv, G_PV, PV_SH, PV_MASK)):
            H = np.full((len(cs),) + sh, -1024, np.int16); M = np.zeros((len(cs),) + sh, bool)
            pres = np.zeros(len(cs), bool); nv = np.zeros(len(cs), np.int32); C = np.full((len(cs), 3), np.nan, np.float32)
            for k, (c, n) in enumerate(cs):
                nv[k] = n
                if c is None:
                    continue
                H[k] = np.clip(np.rint(sample(hu, raff, c, G, sh, 1, -1024)), -1024, 3071).astype(np.int16)
                L = sample(lab, maff, c, G, sh, 0, 0).astype(np.uint8)
                M[k] = np.isin(L, masks[k]); pres[k] = True; C[k] = c
            res.update({f"{tag}_hu": H, f"{tag}_m": np.packbits(M.reshape(len(cs), -1), 1), f"{tag}_present": pres,
                        f"{tag}_nvox": nv, f"{tag}_centre": C})
        tmp = dst.replace(".npz", ".tmp.npz")
        np.savez_compressed(tmp, **res); os.rename(tmp, dst)
        return cid, f"ok sp{int(res['sp_present'].sum())} pv{int(res['pv_present'].sum())}", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return cid, "ERR " + repr(e)[:200] + " " + traceback.format_exc().splitlines()[-2][:120], time.time() - t0


if __name__ == "__main__":
    ids, out, npr = sys.argv[1], sys.argv[2], int(sys.argv[3]); mdirs = sys.argv[4:]
    os.makedirs(f"{out}/crops", exist_ok=True)
    todo = [l.strip() for l in open(ids) if l.strip()]
    todo = [c for c in todo if not os.path.exists(f"{out}/crops/{c}.npz")]
    print(f"{len(todo)} to do", flush=True)
    n = 0; t0 = time.time()
    with Pool(npr, maxtasksperchild=50) as pool, open(f"{out}/status.tsv", "a") as fh:
        for cid, st, dt in pool.imap(one, [(c, out, mdirs) for c in todo], chunksize=1):
            fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            if os.path.exists(f"{out}/STOP"):
                print("STOP", flush=True); pool.terminate(); break
    print("CROP_DONE", flush=True)
