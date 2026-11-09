"""spine_ft / pelvis_ft crops from the existing 3 mm prep (outputs/prep/{id}.npz: hu (z,y,x) on the same canonical 3 mm
grid as the TS Dataset297 mask outputs/masks/{id}.nii.gz (x,y,z)). No raw-CT reads (the CT-rate HDD is saturated).
Centres = frac15_crop.centres (3 mm TS labels), crops = integer slices on the 3 mm grid (no interpolation), padded with
-1024 / 0. The FT script upsamples x2 (trilinear HU, nearest mask) on the GPU -> 1.5 mm-spaced encoder input.
  spine  [10, 20, 24, 20] (z,y,x) T8..L5 ; mask = that vertebra
  pelvis [7, 24, 24, 24] acet_r, acet_l, pubic, sac_r, sac_l, fem_r, fem_l ; mask = unit bone(s)
usage: crop_prep.py ids.txt prep_dir out_pack.npz n_proc mask_dir [mask_dir ...]"""
import sys, os, time, numpy as np, nibabel as nib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frac15_crop as F

SPS, PVS = (20, 24, 20), (24, 24, 24)


def cut(vol, c, sh, fill):
    """vol (z,y,x); c centre index (z,y,x) float; integer crop of shape sh centred at c."""
    lo = np.floor(np.asarray(c) - np.asarray(sh) / 2 + 0.5).astype(int)
    out = np.full(sh, fill, vol.dtype)
    s = [slice(max(l, 0), min(l + n, d)) for l, n, d in zip(lo, sh, vol.shape)]
    if any(x.stop <= x.start for x in s):
        return out, False
    d = tuple(slice(x.start - l, x.stop - l) for x, l in zip(s, lo))
    out[d] = vol[tuple(s)]
    return out, True


def one(args):
    cid, prep, mdirs = args
    try:
        pp = f"{prep}/{cid}.npz"
        if not os.path.exists(pp):
            return cid, None, "missing"
        z = np.load(pp); hu = z["hu"]; m = None; first = None
        for d in mdirs:  # first mask on the prep grid; else resample prep HU onto the first mask's grid
            if os.path.exists(f"{d}/{cid}.nii.gz"):
                mm = nib.load(f"{d}/{cid}.nii.gz"); first = first or mm
                if np.allclose(z["affine"], mm.affine, atol=1e-3) and hu.shape == mm.shape[::-1]:
                    m = mm; break
        stat = "ok"
        if m is None:
            if first is None:
                return cid, None, "no mask"
            m = first; stat = "ok_resampled"
            from scipy import ndimage
            T = np.linalg.inv(z["affine"]) @ m.affine  # mask (x,y,z) index -> prep (x,y,z) index
            g = np.indices(m.shape[::-1]).reshape(3, -1)[::-1]  # (x,y,z) of every (z,y,x) voxel of the mask grid
            q = (T[:3, :3] @ g + T[:3, 3:])[::-1]  # prep (z,y,x) index
            hu = ndimage.map_coordinates(hu.astype(np.float32), q, order=1, cval=-1024, prefilter=False)
            hu = np.rint(hu).astype(np.int16).reshape(m.shape[::-1])
        lab = np.asarray(m.dataobj).astype(np.uint8); A = m.affine
        labz = lab.transpose(2, 1, 0)
        sp, pv = F.centres(lab, A)
        Ai = np.linalg.inv(A)
        r = {}
        for tag, cs, sh, masks in (("sp", sp, SPS, [(l,) for l in F.SP_LV]), ("pv", pv, PVS, F.PV_MASK)):
            H = np.full((len(cs),) + sh, -1024, np.int16); M = np.zeros((len(cs),) + sh, np.uint8)
            pres = np.zeros(len(cs), bool); nv = np.zeros(len(cs), np.int32)
            for k, (c, n) in enumerate(cs):
                nv[k] = n
                if c is None:
                    continue
                ijk = (Ai @ np.r_[c, 1])[:3][::-1]  # x,y,z index -> z,y,x
                H[k], ok = cut(hu, ijk, sh, np.int16(-1024))
                L, _ = cut(labz, ijk, sh, np.uint8(0))
                M[k] = np.isin(L, masks[k]); pres[k] = ok
            r.update({f"{tag}_hu": H, f"{tag}_m": M, f"{tag}_present": pres, f"{tag}_nvox": nv})
        return cid, r, stat
    except Exception as e:  # noqa: BLE001
        return cid, None, "ERR " + repr(e)[:200]


if __name__ == "__main__":
    ids, prep, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); mdirs = sys.argv[5:]
    todo = [l.strip() for l in open(ids) if l.strip() and os.path.exists(f"{prep}/{l.strip()}.npz")]
    print(f"{len(todo)} with prep", flush=True)
    keep, R, st = [], {}, []
    t0 = time.time()
    with Pool(npr) as pool:
        for i, (cid, r, s) in enumerate(pool.imap(one, [(c, prep, mdirs) for c in todo], chunksize=8)):
            st.append((cid, s))
            if r is not None:
                keep.append(cid)
                for k, v in r.items():
                    R.setdefault(k, []).append(v)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, ids=np.array(keep), **{k: np.stack(v) for k, v in R.items()})
    with open(out.replace(".npz", "_status.tsv"), "w") as fh:
        fh.writelines(f"{c}\t{s}\n" for c, s in st)
    print("PACK_DONE", len(keep), f"{time.time() - t0:.0f}s", flush=True)
