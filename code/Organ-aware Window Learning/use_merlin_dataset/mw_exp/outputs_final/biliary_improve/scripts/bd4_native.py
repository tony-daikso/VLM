"""BD4 pilot: native-resolution porta-hepatis / CBD features from the ORIGINAL CT (Mac Passport drive).
The 3 mm TotalSegmentator 'total' mask (x,y,z, RAS affine) gives the ROI; the native CT is cropped by world
coordinates (affines handle LAS/RAS), resampled to 1 mm isotropic along the native axes (order 1) and the mask is
nearest-neighbour resampled onto that grid. Features (mm / ml, resolution independent):
  hdl_*  hepatoduodenal-ligament corridor (unlabelled, <=24 mm from portal vein, <=30 mm from liver or pancreas):
         largest fluid (-5..30 HU, 0.7 mm smoothed) component vol / max diameter (2*max EDT) / cranio-caudal length;
         'cbd' = component with the largest max diameter among components >=10 mm long (CBD candidate)
  ph_*   pancreatic-head corridor (<=9 mm from pancreas and <=15 mm from duodenum): same
  tube_* multi-scale Frangi dark tubes (sigma 1/1.5/2.5/4 mm, black ridges) in corridor and in hilar liver
         (<=30 mm from PV): volume of dark tubular voxels, volume-weighted mean scale, max scale
usage: bd4_native.py ids.txt ct_dir mask_dir out.csv nproc"""
import sys, os, time, numpy as np, pandas as pd, nibabel as nib, warnings
from nibabel.processing import resample_from_to
from multiprocessing import Pool
from scipy import ndimage
warnings.filterwarnings("ignore")
C = dict(liver=5, panc=7, duod=19, pv=64, ivc=63, gb=4)


def comps(mask, zooms_mm=1.0, min_len=0):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return []
    out = []
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        c = lab[sl] == i
        if c.sum() < 8:
            continue
        e = ndimage.distance_transform_edt(np.pad(c, 1))
        zl = c.any(axis=(0, 1)).sum()            # native axis 2 ~ cranio-caudal
        out.append(dict(vol=c.sum() / 1000.0, dmax=2 * e.max(), zlen=float(zl)))
    return out


def summarize(cl, pre, d):
    if not cl:
        for k in ("vol", "dmax", "zlen", "cbd_dmax", "cbd_len", "n"):
            d[f"{pre}_{k}"] = 0.0
        return
    big = max(cl, key=lambda r: r["vol"])
    d[f"{pre}_vol"], d[f"{pre}_dmax"], d[f"{pre}_zlen"] = big["vol"], big["dmax"], big["zlen"]
    tub = [r for r in cl if r["zlen"] >= 10]
    cb = max(tub, key=lambda r: r["dmax"]) if tub else {"dmax": 0.0, "zlen": 0.0}
    d[f"{pre}_cbd_dmax"], d[f"{pre}_cbd_len"] = cb["dmax"], cb["zlen"]
    d[f"{pre}_n"] = float(len(cl))


def feat(args):
    sid, ctd, mdir = args
    t0 = time.time(); d = {"study_id": sid}
    try:
        mi = nib.load(f"{mdir}/{sid}.nii.gz"); m3 = np.asarray(mi.dataobj).astype(np.uint8); A3 = mi.affine
        PV = m3 == C["pv"]; L = m3 == C["liver"]; P = m3 == C["panc"]
        if PV.sum() < 20 or L.sum() < 500:
            d["n_err"] = "nopv"; return d
        dpv = ndimage.distance_transform_edt(~PV)
        roi = PV | P | (L & (dpv <= 12)) | ((m3 == C["duod"]) & (dpv <= 15))
        ix = np.argwhere(roi); lo3, hi3 = ix.min(0) - 3, ix.max(0) + 3
        img = nib.load(f"{ctd}/{sid}.nii.gz"); An = img.affine; zooms = np.array(img.header.get_zooms()[:3], float)
        corners = np.array([[a, b, c, 1] for a in (lo3[0], hi3[0]) for b in (lo3[1], hi3[1]) for c in (lo3[2], hi3[2])], float)
        nat = (np.linalg.inv(An) @ (A3 @ corners.T)).T[:, :3]
        shp = np.array(img.shape[:3]); lo = np.clip(np.floor(nat.min(0)).astype(int) - 2, 0, shp); hi = np.clip(np.ceil(nat.max(0)).astype(int) + 3, 0, shp)
        if np.any(hi - lo < 3):
            d["n_err"] = "crop_outside"; return d
        h = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
        t_load = time.time() - t0
        Ac = An.copy(); Ac[:3, 3] = (An @ np.r_[lo, 1])[:3]
        # 1 mm isotropic along native axes
        h = ndimage.zoom(h, zooms, order=1)
        A1 = Ac.copy(); A1[:3, :3] = Ac[:3, :3] / zooms
        m = np.asarray(resample_from_to(nib.Nifti1Image(m3, A3), (h.shape, A1), order=0).dataobj).astype(np.uint8)
    except Exception as e:  # noqa: BLE001
        d["n_err"] = f"load:{str(e)[:60]}"; return d
    d.update(n_dx=zooms[0], n_dz=zooms[2], n_shape=int(np.prod(h.shape)))
    hs = ndimage.gaussian_filter(np.clip(h, -200, 400), 0.7)
    PVn, Ln, Pn, Dn = m == C["pv"], m == C["liver"], m == C["panc"], m == C["duod"]
    if PVn.sum() < 100:
        d["n_err"] = "nopv_native"; return d
    edt = lambda x: ndimage.distance_transform_edt(~x) if x.any() else np.full(x.shape, 999.0)
    dpv, dl, dp, dd = edt(PVn), edt(Ln), edt(Pn), edt(Dn)
    free = m == 0
    lmed = float(np.median(hs[Ln])) if Ln.sum() > 100 else 100.0
    d["n_liver_hu"] = lmed
    fluid = (hs >= -5) & (hs <= 30)
    hdl = free & (dpv <= 24) & ((dl <= 30) | (dp <= 30))
    ph = (free | Pn) & (dp <= 9) & (dd <= 15)
    hil = Ln & (dpv <= 30)
    d["n_hdl_ml"] = hdl.sum() / 1000.0
    summarize(comps(hdl & fluid), "n_hdl", d)
    summarize(comps(hdl & (hs >= -5) & (hs <= 20)), "n_hdlw", d)
    summarize(comps(ph & fluid), "n_ph", d)
    d["n_hdl_fluid_ml"] = (hdl & fluid).sum() / 1000.0
    d["n_hil_dark_ml"] = (hil & (hs < lmed - 35) & (hs > -20)).sum() / 1000.0
    # Frangi dark tubes (1 mm grid)
    try:
        from skimage.filters import frangi
        x = (np.clip(hs, -50, 200) + 50) / 250.0
        best = np.zeros(x.shape, np.float32); scale = np.zeros(x.shape, np.float32)
        for s in (1.0, 1.5, 2.5, 4.0):
            v = frangi(x, sigmas=(s,), black_ridges=True, alpha=0.5, beta=0.5, gamma=None).astype(np.float32)
            up = v > best; best[up] = v[up]; scale[up] = s
        dark = (hs < lmed - 30) & (hs > -20)
        for tag, reg in (("hdl", hdl), ("hil", hil), ("ph", ph)):
            for thr in (0.02, 0.06):
                t = reg & dark & (best > thr)
                k = f"n_tube_{tag}_t{int(thr * 100):02d}"
                d[f"{k}_ml"] = t.sum() / 1000.0
                d[f"{k}_mscale"] = float(scale[t].mean()) if t.any() else 0.0
                d[f"{k}_xscale"] = float(scale[t].max()) if t.any() else 0.0
    except Exception as e:  # noqa: BLE001
        d["n_err_tube"] = str(e)[:60]
    d["t_load"] = t_load; d["t_total"] = time.time() - t0
    return d


if __name__ == "__main__":
    ids, ctd, mdir, out, npr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    L_ = [l.strip() for l in open(ids) if l.strip()]
    res = []
    if os.path.exists(out):
        old = pd.read_csv(out); res = old.to_dict("records"); seen = set(old.study_id); L_ = [s for s in L_ if s not in seen]
    with Pool(npr, maxtasksperchild=20) as p:
        for i, r in enumerate(p.imap_unordered(feat, [(s, ctd, mdir) for s in L_], chunksize=1)):
            res.append(r)
            if (i + 1) % 20 == 0:
                pd.DataFrame(res).to_csv(out, index=False); print(i + 1, "done", flush=True)
    pd.DataFrame(res).to_csv(out, index=False); print("BD4_DONE", len(res), flush=True)
