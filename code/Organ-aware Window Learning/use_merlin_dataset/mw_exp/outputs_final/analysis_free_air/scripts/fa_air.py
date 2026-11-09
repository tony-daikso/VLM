"""Extraluminal-air features from 3 mm prep HU + TotalSegmentator(total, fast) label map.
Usage: fa_air.py ids.txt prep_dir mask_dir out.csv nproc   (prep hu is [z,y,x]; mask nifti is [x,y,z] RAS -> transpose(2,1,0); +y = anterior)"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage as nd
ids, prep, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
LUNG = [10, 11, 12, 13, 14, 15, 16]          # lung lobes, esophagus, trachea
GUT = [6, 18, 19, 20]                       # stomach, small bowel, duodenum, colon
BONE = list(range(25, 51)) + list(range(69, 79)) + list(range(91, 118))
B3 = nd.generate_binary_structure(3, 1)
def cc_stats(m):
    lab, n = nd.label(m, B3)
    if n == 0: return 0, 0, lab, np.zeros(1, int)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    return n, int(sz.max()), lab, sz
def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz"); hu = z["hu"].astype(np.float32)
        ts = np.asanyarray(nib.load(f"{mdir}/{sid}.nii.gz").dataobj).transpose(2, 1, 0)
    except Exception:
        return None
    if ts.shape != hu.shape: return {"study_id": sid, "err": 1}
    d = {"study_id": sid, "err": 0, "nz": hu.shape[0]}
    # body: largest >-500 component, per-slice hole fill, 2-vox erosion
    b = hu > -500
    lab, n = nd.label(b, B3)
    if n == 0: return d
    sz = np.bincount(lab.ravel()); sz[0] = 0; b = lab == sz.argmax()
    for k in range(b.shape[0]): b[k] = nd.binary_fill_holes(b[k])
    body = nd.binary_erosion(b, iterations=2)
    deep = nd.binary_erosion(b, iterations=7)            # >=21 mm from skin: excludes most subcutaneous air
    gut = np.isin(ts, GUT); gut1 = nd.binary_dilation(gut, iterations=1); gut3 = nd.binary_dilation(gut, iterations=3)
    lung = nd.binary_dilation(np.isin(ts, LUNG), iterations=2)
    liver = ts == 5
    d["lung_vox"] = int(np.isin(ts, [10, 11, 12, 13, 14]).sum()); d["liver_vox"] = int(liver.sum())
    d["gut_vox"] = int(gut.sum()); d["body_vox"] = int(body.sum())
    for t in (-900, -600):
        air = (hu < t) & body
        tag = str(-t)
        d[f"air{tag}_total"] = int(air.sum())
        d[f"air{tag}_inlumen"] = int((air & gut1).sum())
        d[f"air{tag}_ring"] = int((air & gut3 & ~gut1 & ~lung).sum())       # 3-9 mm from gut mask: mask leak or peri-bowel air
        ex = air & ~gut3 & ~lung
        d[f"air{tag}_extra"] = int(ex.sum())
        exd = ex & deep
        d[f"air{tag}_extra_deep"] = int(exd.sum())
        n, mx, lb, s = cc_stats(exd)
        d[f"air{tag}_deep_ncc"] = n; d[f"air{tag}_deep_maxcc"] = mx
        d[f"air{tag}_deep_small"] = int(s[(s > 0) & (s < 2000)].sum())    # excludes huge lung/bowel-like blobs
        d[f"air{tag}_deep_ncc_ge2"] = int(((s >= 2) & (s < 2000)).sum())
        if liver.sum() > 100:
            lz, ly, lx = np.nonzero(liver)
            near = nd.binary_dilation(liver, iterations=5) & ~liver
            ant = np.zeros_like(liver); ant[:, int(np.percentile(ly, 50)):, :] = True
            d[f"air{tag}_periliver"] = int((exd & near).sum())
            d[f"air{tag}_antliver"] = int((exd & near & ant).sum())
        # anterior third of the body, below lung
        ys = np.nonzero(body.any((0, 2)))[0]
        if len(ys):
            ycut = ys.min() + 0.66 * (ys.max() - ys.min())
            antm = np.zeros_like(body); antm[:, int(ycut):, :] = True
            d[f"air{tag}_ant_deep"] = int((exd & antm).sum())
        # retroperitoneal proxy: posterior half, around kidneys/duodenum/pancreas
        rp = nd.binary_dilation(np.isin(ts, [2, 3, 7, 19]), iterations=4) & ~gut1
        d[f"air{tag}_retroperi"] = int((air & rp & ~lung).sum())
    # surgery proxy: metal-dense voxels outside bone/gut lumen
    met = (hu > 1500) & body & ~nd.binary_dilation(np.isin(ts, BONE), iterations=1) & ~gut1
    d["metal_vox"] = int(met.sum()); d["metal_ncc"] = int(nd.label(met, B3)[1])
    d["hu_max_soft"] = float(hu[body & ~np.isin(ts, BONE)].max()) if body.any() else np.nan
    return d
COLS = "study_id,err,nz,lung_vox,liver_vox,gut_vox,body_vox,air900_total,air900_inlumen,air900_ring,air900_extra,air900_extra_deep,air900_deep_ncc,air900_deep_maxcc,air900_deep_small,air900_deep_ncc_ge2,air900_periliver,air900_antliver,air900_ant_deep,air900_retroperi,air600_total,air600_inlumen,air600_ring,air600_extra,air600_extra_deep,air600_deep_ncc,air600_deep_maxcc,air600_deep_small,air600_deep_ncc_ge2,air600_periliver,air600_antliver,air600_ant_deep,air600_retroperi,metal_vox,metal_ncc,hu_max_soft".split(",")
if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    done = set(pd.read_csv(out).study_id) if os.path.exists(out) else set()
    L = [s for s in L if s not in done]          # resumable; ids file is in priority order
    res = []
    with Pool(npr) as p:
        for i, r in enumerate(p.imap(feat, L, chunksize=2)):
            if r is not None: res.append(r)
            if len(res) >= 100 or i == len(L) - 1:
                pd.DataFrame(res).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False); res = []
                print("progress", i + 1, "of", len(L), flush=True)
    print("done", flush=True)
