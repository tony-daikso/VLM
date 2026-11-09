"""GS2: native-resolution GB features from the ORIGINAL thin-slice CT.
The 3 mm TS --fast GB mask (prep npz 'units', RAS affine) is resampled (nearest neighbour, nibabel resample_from_to)
onto a crop of the native grid (LAS etc. handled by the affines).
usage: gs2_native.py ids.txt ct_dir prep_dir out.csv nproc"""
import sys, time, numpy as np, pandas as pd, nibabel as nib
from nibabel.processing import resample_from_to
from multiprocessing import Pool
from scipy import ndimage, stats

def cc_max(b, vv):
    lab, n = ndimage.label(b)
    return float(np.bincount(lab.ravel())[1:].max() * vv) if n else 0.0

def feat(args):
    sid, ctd, prd = args
    t0 = time.time()
    try:
        z = np.load(f"{prd}/{sid}.npz"); names = list(z["unit_names"]); gi = names.index("gallbladder") + 1
        m3 = (z["units"] == gi).transpose(2, 1, 0)  # (z,y,x) -> (x,y,z) on the RAS 3 mm grid
        A3 = z["affine"]
    except Exception as e:
        return {"study_id": sid, "n_err": f"prep:{e}"}
    d = {"study_id": sid, "n_gb3_vox": int(m3.sum())}
    if m3.sum() < 5:
        return d
    try:
        img = nib.load(f"{ctd}/{sid}.nii.gz"); An = img.affine; zooms = np.array(img.header.get_zooms()[:3], float)
        # crop box: 3 mm GB bbox + 3 voxels (9 mm) margin -> world corners -> native index box
        ix = np.argwhere(m3); lo3, hi3 = ix.min(0) - 3, ix.max(0) + 3
        corners = np.array([[a, b, c, 1] for a in (lo3[0], hi3[0]) for b in (lo3[1], hi3[1]) for c in (lo3[2], hi3[2])], float)
        nat = (np.linalg.inv(An) @ (A3 @ corners.T)).T[:, :3]
        shp = np.array(img.shape[:3]); lo = np.clip(np.floor(nat.min(0)).astype(int) - 2, 0, shp); hi = np.clip(np.ceil(nat.max(0)).astype(int) + 3, 0, shp)
        if np.any(hi - lo < 3):
            d["n_err"] = "crop_outside"; return d
        raw = np.asarray(img.dataobj.get_unscaled() if hasattr(img.dataobj, "get_unscaled") else img.dataobj)
        sl, it = img.dataobj.slope, img.dataobj.inter
        h = raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32); del raw
        if sl is not None and not np.isnan(sl): h = h * float(sl) + float(it if it is not None and not np.isnan(it) else 0)
        t_load = time.time() - t0
        Ac = An.copy(); Ac[:3, 3] = (An @ np.r_[lo, 1])[:3]
        m = np.asarray(resample_from_to(nib.Nifti1Image(m3.astype(np.uint8), A3), (h.shape, Ac), order=0).dataobj) > 0
    except Exception as e:
        d["n_err"] = f"ct:{e}"; return d
    vv = float(np.prod(zooms)); d.update(n_dx=zooms[0], n_dz=zooms[2], n_vox=int(m.sum()), n_vol_ml=float(m.sum() * vv / 1000))
    if m.sum() < 20:
        d["n_err"] = "tiny_native"; return d
    # dilated mask (3 mm) and interior (>= 3 mm from the GB boundary: excludes the wall rim / partial volume)
    md = ndimage.distance_transform_edt(~m, sampling=zooms) <= 3.0
    din = ndimage.distance_transform_edt(m, sampling=zooms)
    inn = din >= 3.0
    if inn.sum() < 20: inn = din >= 1.5
    for tag, r in (("m", m), ("md", md)):
        v = h[r]
        d[f"n_{tag}_max"] = float(v.max()); d[f"n_{tag}_p99"] = float(np.percentile(v, 99)); d[f"n_{tag}_p999"] = float(np.percentile(v, 99.9))
        for t in (130, 200, 400):
            b = r & (h > t)
            d[f"n_{tag}_ml{t}"] = float(b.sum() * vv / 1000)  # volumes in ml (resolution-independent)
            d[f"n_{tag}_cc{t}"] = cc_max(b, vv)               # largest connected hyperdense component, mm^3
    # bile reference from the interior
    vin = h[inn] if inn.sum() >= 20 else h[m]
    med = float(np.median(vin)); d["n_bile_med"] = med; d["n_in_std"] = float(vin.std()); d["n_in_skew"] = float(stats.skew(vin))
    # smoothing (sigma 1 mm) to suppress thin-slice noise before defect / blob detection
    sig1 = 1.0 / zooms
    hs = ndimage.gaussian_filter(np.clip(h, -300, 1500), sig1)
    w = hs[inn] if inn.sum() >= 20 else hs[m]
    d["n_in_sm_std"] = float(w.std())
    hyp = inn & (hs < med - 20)
    d["n_hypo_ml"] = float(hyp.sum() * vv / 1000); d["n_hypo_frac"] = float(hyp.sum() / max(inn.sum(), 1)); d["n_hypo_cc"] = cc_max(hyp, vv)
    hyp2 = inn & (hs < med - 40); d["n_hypo40_cc"] = cc_max(hyp2, vv)
    hpr = inn & (hs > med + 40); d["n_hyper40_cc"] = cc_max(hpr, vv); d["n_hyper40_ml"] = float(hpr.sum() * vv / 1000)
    # DoG blob detection at 2 / 3 / 4 mm
    hc = np.clip(h, -300, 1500)
    for s in (2.0, 3.0, 4.0):
        dog = ndimage.gaussian_filter(hc, s / zooms) - ndimage.gaussian_filter(hc, 1.6 * s / zooms)
        q = dog[inn] if inn.sum() >= 20 else dog[m]
        k = int(s)
        d[f"n_dog{k}_max"] = float(q.max()); d[f"n_dog{k}_min"] = float(q.min())
        d[f"n_dog{k}_p99"] = float(np.percentile(q, 99)); d[f"n_dog{k}_p1"] = float(np.percentile(q, 1))
        d[f"n_dog{k}_ml_pos15"] = float((q > 15).sum() * vv / 1000); d[f"n_dog{k}_ml_neg15"] = float((q < -15).sum() * vv / 1000)
    # dependent layering: world A (RAS y) coordinate; supine -> posterior (low y) is dependent
    ijk = np.argwhere(inn if inn.sum() >= 60 else m)
    yw = (Ac[1, :3] @ ijk.T) + Ac[1, 3]
    vals = hs[tuple(ijk.T)]; a, b = np.percentile(yw, [33.3, 66.7])
    post, ant = vals[yw <= a], vals[yw >= b]
    if post.size > 5 and ant.size > 5:
        d["n_dep_diff"] = float(np.median(post) - np.median(ant)); d["n_dep_p90diff"] = float(np.percentile(post, 90) - np.percentile(ant, 90))
        d["n_dep_p10diff"] = float(np.percentile(post, 10) - np.percentile(ant, 10))
    d["t_load"] = t_load; d["t_total"] = time.time() - t0
    return d

if __name__ == "__main__":
    ids, ctd, prd, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    import os
    L = [l.strip() for l in open(ids) if l.strip()]
    res = []
    if os.path.exists(out):  # resume: keep finished rows, redo only missing ids
        old = pd.read_csv(out); res = old.to_dict("records"); seen = set(old.study_id); L = [s for s in L if s not in seen]
        print("resume: kept", len(res), "todo", len(L), flush=True)
    with Pool(npr, maxtasksperchild=20) as p:
        for i, r in enumerate(p.imap_unordered(feat, [(s, ctd, prd) for s in L], chunksize=1)):
            res.append(r)
            if (i + 1) % 25 == 0:
                pd.DataFrame(res).to_csv(out, index=False); print(i + 1, "done", flush=True)
    pd.DataFrame(res).to_csv(out, index=False); print("done", len(res), "rows; this run", len(L), flush=True)
