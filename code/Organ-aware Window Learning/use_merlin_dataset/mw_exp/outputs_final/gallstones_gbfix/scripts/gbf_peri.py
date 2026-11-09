"""Task A follow-up: peri-GB search for GB-present cases (3 mm TS GB >= 100 vox) -- the two failure modes seen in the PNG
review: (1) stone just outside the mask at the neck / cystic duct, (2) under-segmented distended GB with stones in the
unsegmented (dependent) part.  Train/val ids only; no labels read.
Features (liver-relative threshold thr = max(130, liver_med + 50)):
  pr10 / pr25 : region = outside GB, 0-10 mm / 10-25 mm from the mask, liver depth <= 3 mm, minus bone(+1)/vessels/other organs
                (same blob features as gbf_fossa.region_feats)
  nk20        : 20 mm ball around the GB neck point (GB voxel nearest the porta / portal vein), outside the mask, same exclusions
  grow        : geodesic growth of the GB through fluid-density voxels (-10..45 HU, not excluded, liver depth <= 3 mm), 10 x 3 mm;
                ext_vol = grown - GB (mL), ext_frac = ext/GB, blobs > thr inside the grown extension dilated 1 vox
usage: gbf_peri.py ids.txt out.csv nproc prepdir maskmap.tsv tmpdir"""
import sys, os, subprocess
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
sys.argv_saved = list(sys.argv)
IDS, OUT, NP, PREP, MM, TMP = sys.argv[1:7]
# reuse gbf_fossa helpers (argv-independent parts)
sys.argv = [sys.argv[0], "x", "x", "1", PREP, MM, "x", TMP]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gbf_fossa as G
NP = int(NP)
ST26 = np.ones((3, 3, 3), bool); SP = 3.0


def run(sid):
    try:
        fp, tmp = G.get_prep(sid)
        if fp is None: return dict(study_id=sid, err="no_prep")
        z = np.load(fp); hu = z["hu"].astype(np.float32); names = list(z["unit_names"]); gb = z["units"] == names.index("gallbladder") + 1
        if tmp: os.remove(fp)
        m = np.asanyarray(nib.load(G.MMAP[sid]).dataobj).astype(np.int16).transpose(2, 1, 0)
        f = dict(study_id=sid, gb3_vox=int(gb.sum()))
        if gb.sum() < 100: return f
        if (m == G.LIV).sum() < 1000: f["err"] = "no_liver"; return f
        gi0 = np.argwhere(gb); lo = np.maximum(gi0.min(0) - 20, 0); hi = gi0.max(0) + 21
        sl = tuple(slice(a, b) for a, b in zip(lo, hi)); hu, m, gb = hu[sl], m[sl], gb[sl]
        L = m == G.LIV
        if L.sum() < 50: f["err"] = "no_liver_crop"; return f
        Lout = ndimage.distance_transform_edt(~L) * SP; Ld = ndimage.distance_transform_edt(L) * SP
        lmed = float(np.median(hu[L & (Ld > 9)])) if (L & (Ld > 9)).sum() > 100 else float(np.median(hu[L]))  # liver within the crop
        thr = max(130.0, lmed + 50.0); f["p_liver_med"] = lmed
        pv = (m == G.PV) & (Lout <= 10); Ppv = np.array(ndimage.center_of_mass(pv)) if pv.sum() >= 5 else None
        excl = (ndimage.binary_dilation(np.isin(m, G.BONE), iterations=1) | np.isin(m, G.VESS + G.OTH + [G.DUO])) & ~gb
        Dg = ndimage.distance_transform_edt(~gb) * SP
        cen = np.array(ndimage.center_of_mass(gb))
        base = (Ld <= 3) & ~excl & ~gb
        f.update(G.region_feats(hu, m, base & (Dg > 0) & (Dg <= 10), cen, Ppv, Lout, Ld, "pr10", thr))
        f.update(G.region_feats(hu, m, base & (Dg > 10) & (Dg <= 25), cen, Ppv, Lout, Ld, "pr25", thr))
        gi = np.argwhere(gb)
        if Ppv is not None:
            nk = gi[np.argmin(((gi - Ppv) ** 2).sum(1))].astype(float)
            f["neck_to_pv_mm"] = float(np.linalg.norm(nk - Ppv) * SP)
        else:
            nk = gi[np.argmin(gi[:, 1])].astype(float)
        zz, yy, xx = np.ogrid[:hu.shape[0], :hu.shape[1], :hu.shape[2]]
        dN = np.sqrt((zz - nk[0]) ** 2 + (yy - nk[1]) ** 2 + (xx - nk[2]) ** 2) * SP
        f.update(G.region_feats(hu, m, base & (dN <= 20), nk, Ppv, Lout, Ld, "nk20", thr))
        # under-segmentation: grow GB through fluid
        fluid = (hu >= -10) & (hu <= 45) & ~excl & (Ld <= 3)
        g = gb.copy()
        for _ in range(10):
            g2 = ndimage.binary_dilation(g, iterations=1) & (fluid | gb)
            if (g2 == g).all(): break
            g = g2
        ext = g & ~gb
        f["grow_ext_ml"] = float(ext.sum() * SP ** 3 / 1000); f["grow_ext_frac"] = float(ext.sum() / gb.sum())
        if ext.any():
            ed = ndimage.binary_dilation(ext, iterations=1) & ~gb & ~excl
            v = hu[ed]; f["grow_ext_max"] = float(v.max()); f["grow_ext_n_thr"] = int((v > thr).sum())
            f["grow_ext_dep_frac"] = float((np.argwhere(ext)[:, 1] < np.percentile(gi[:, 1], 33)).mean())
        return f
    except Exception as e:
        return dict(study_id=sid, err=str(e)[:200])


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    L = [l.strip() for l in open(IDS) if l.strip()]
    old = []
    for fo in (OUT, OUT + ".part"):
        if os.path.exists(fo) and os.path.getsize(fo) > 0:
            d0 = pd.read_csv(fo); d0 = d0[d0.err.isna()] if "err" in d0 else d0; old.append(d0)
    done = set(pd.concat(old).study_id) if old else set()
    L = [s for s in L if s not in done]; print("todo", len(L), "done", len(done), flush=True)
    R = [r for d0 in old for r in d0.to_dict("records")]
    with Pool(NP) as p:
        for i, r in enumerate(p.imap_unordered(run, L, chunksize=4)):
            R.append(r)
            if (i + 1) % 500 == 0: print(i + 1, len(L), flush=True); pd.DataFrame(R).drop_duplicates("study_id", keep="last").to_csv(OUT + ".part", index=False)
    pd.DataFrame(R).drop_duplicates("study_id", keep="last").to_csv(OUT, index=False); print("DONE", len(R), flush=True)
