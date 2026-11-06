"""GS1: extra 3 mm GB intensity/shape/texture features from prep npz (hu + units; RAS, array z,y,x).
usage: gs1_feat.py ids.txt prep_dir out.csv nproc"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage, stats
ids, prep, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def basic(v, p):
    d = {}
    if v.size < 5: return d
    med = np.median(v)
    d[p+"med"] = float(med); d[p+"std"] = float(v.std()); d[p+"iqr"] = float(np.subtract(*np.percentile(v, [75, 25])))
    d[p+"skew"] = float(stats.skew(v)); d[p+"kurt"] = float(stats.kurtosis(v))
    d[p+"f_lo20"] = float((v < med - 20).mean()); d[p+"f_lo10"] = float((v < med - 10).mean())
    d[p+"f_hi40"] = float((v > med + 40).mean()); d[p+"f_hi20"] = float((v > med + 20).mean())
    d[p+"p1"] = float(np.percentile(v, 1)); d[p+"max_minus_med"] = float(v.max() - med)
    return d
def feat(sid):
    try:
        z = np.load(f"{prep}/{sid}.npz")
        names = list(z["unit_names"]); u = z["units"]; gi = names.index("gallbladder") + 1
        m = u == gi
    except Exception:
        return None
    d = {"study_id": sid, "g1_vox": int(m.sum())}
    if m.sum() < 5: return d
    hu = z["hu"].astype(np.float32)
    sl = ndimage.find_objects(m.astype(np.uint8))[0]
    sl = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in sl)
    m = m[sl]; h = hu[sl]
    lab, n = ndimage.label(m); sz = np.bincount(lab.ravel())[1:]
    d["g1_ncc"] = int(n); m = lab == (np.argmax(sz) + 1)  # largest component
    d["g1_lcc_vox"] = int(m.sum())
    v = h[m]; d.update(basic(v, "g1_"))
    inn = ndimage.binary_erosion(m)  # exclude wall rim
    d["g1_in_vox"] = int(inn.sum())
    if inn.sum() >= 10: d.update(basic(h[inn], "g1_in_"))
    # shape
    c = np.argwhere(m).astype(float) * 3.0
    if len(c) >= 10:
        ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]; ev = np.sqrt(np.maximum(ev, 1e-6))
        d["g1_ax1"] = float(ev[0]); d["g1_elong"] = float(ev[0] / ev[1]); d["g1_flat"] = float(ev[1] / ev[2])
        faces = sum(int(np.abs(np.diff(m.astype(np.int8), axis=a)).sum()) for a in range(3)) * 9.0
        V = m.sum() * 27.0
        d["g1_sphericity"] = float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / max(faces, 1))
        d["g1_extent"] = float(m.sum() / np.prod([s.stop - s.start for s in ndimage.find_objects(m.astype(np.uint8))[0]]))
        # dependent layering: axis1 = y (RAS: higher index = anterior); supine -> posterior is dependent
        y = np.argwhere(m)[:, 1]; lo, hi = np.percentile(y, [33.3, 66.7])
        vv = h[m]
        post, ant = vv[y <= lo], vv[y >= hi]
        if post.size >= 3 and ant.size >= 3:
            d["g1_dep_diff"] = float(np.median(post) - np.median(ant)); d["g1_dep_p90diff"] = float(np.percentile(post, 90) - np.percentile(ant, 90))
    # local contrast (DoG ~ 3-7.5 mm) inside the GB interior (or whole GB if tiny)
    hh = np.clip(h, -200, 1000)
    dog = ndimage.gaussian_filter(hh, 1.0) - ndimage.gaussian_filter(hh, 2.5)
    r = inn if inn.sum() >= 10 else m
    w = dog[r]; d["g1_dog_max"] = float(w.max()); d["g1_dog_min"] = float(w.min())
    d["g1_dog_n15"] = int((w > 15).sum()); d["g1_dog_nneg15"] = int((w < -15).sum())
    # local median contrast
    mf = ndimage.median_filter(hh, size=5); lc = (hh - mf)[r]
    d["g1_lc_max"] = float(lc.max()); d["g1_lc_n20"] = int((lc > 20).sum()); d["g1_lc_nneg20"] = int((lc < -20).sum())
    return d
if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=16) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False); print("done", len(res), "of", len(L))
