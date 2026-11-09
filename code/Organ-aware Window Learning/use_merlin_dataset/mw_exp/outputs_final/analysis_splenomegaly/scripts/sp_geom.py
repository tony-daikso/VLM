"""Spleen geometry / FOV / body-size features from prep npz (3 mm iso, z-y-x HU + organ units) and TS total masks.
Usage: sp_geom.py ids.txt prep_dirs(colon-sep) mask_dirs(colon-sep) out.csv nproc   (ids must already be train/val only)
No labels or reports are read here."""
import sys, os, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage
ids, preps, mdirs, out, npr = sys.argv[1], sys.argv[2].split(":"), sys.argv[3].split(":"), sys.argv[4], int(sys.argv[5])
TS = {"spleen": 1, "liver": 5, "L1": 31, "L3": 29, "T12": 32}
VX = 0.027   # mL per 3 mm voxel


def find(dirs, fn):
    for d in dirs:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def feat(sid):
    p = find(preps, sid + ".npz")
    if p is None:
        return None
    try:
        z = np.load(p); hu = z["hu"]; u = z["units"]; names = list(z["unit_names"])
    except Exception:
        return None
    d = {"study_id": sid, "n_slices": hu.shape[0], "fov_z_cm": hu.shape[0] * 0.3}
    sp = u == names.index("spleen") + 1
    d["sp_ml"] = float(sp.sum() * VX)
    d["liver_ml"] = float((u == names.index("liver") + 1).sum() * VX)
    d["kid_ml"] = float(((u == names.index("kidney_left") + 1) | (u == names.index("kidney_right") + 1)).sum() * VX)
    if sp.sum() >= 20:
        lab, n = ndimage.label(sp)
        sz = np.bincount(lab.ravel())[1:]
        main = lab == (sz.argmax() + 1)
        d["sp_ncomp"] = int(n); d["sp_main_frac"] = float(sz.max() / sz.sum())
        d["sp_ncomp_small"] = int(((sz >= 3) & (sz < 0.2 * sz.max())).sum())
        zz = np.where(sp.any((1, 2)))[0]
        d["sp_z0"], d["sp_z1"] = int(zz[0]), int(zz[-1])
        d["sp_cc_cm"] = (zz[-1] - zz[0] + 1) * 0.3
        zm = np.where(main.any((1, 2)))[0]
        d["sp_cc_main_cm"] = (zm[-1] - zm[0] + 1) * 0.3
        d["sp_touch_lo"] = int(zz[0] <= 0); d["sp_touch_hi"] = int(zz[-1] >= hu.shape[0] - 1)
        d["sp_gap_lo"] = int(zz[0]); d["sp_gap_hi"] = int(hu.shape[0] - 1 - zz[-1])
        c = np.argwhere(main).astype(np.float32) * 3.0
        if len(c) > 5:
            cc = c - c.mean(0)
            ev, vec = np.linalg.eigh(np.cov(cc.T))
            pr = cc @ vec[:, ::-1]
            d["sp_pc1_cm"], d["sp_pc2_cm"], d["sp_pc3_cm"] = [float(np.ptp(pr[:, i]) / 10 + 0.3) for i in range(3)]
        yy = np.where(main.any((0, 2)))[0]; xx = np.where(main.any((0, 1)))[0]
        d["sp_ap_cm"] = (yy[-1] - yy[0] + 1) * 0.3; d["sp_tr_cm"] = (xx[-1] - xx[0] + 1) * 0.3
        d["sp_max_axial_cm2"] = float(main.sum((1, 2)).max() * 0.09)
        er = ndimage.binary_erosion(main, iterations=1)
        if er.sum() >= 20:
            d["sp_hu_med"] = float(np.median(hu[er])); d["sp_hu_iqr"] = float(np.subtract(*np.percentile(hu[er], [75, 25])))
        d["sp_zc"] = float(zz.mean())
    try:
        mp = find(mdirs, sid + ".nii.gz")
        m = np.asarray(nib.load(mp).dataobj).transpose(2, 1, 0)
        if m.shape != hu.shape:
            raise ValueError("shape")
        d["ts_sp_ml"] = float((m == TS["spleen"]).sum() * VX)
        for lvl in ("T12", "L1", "L3"):
            vm = m == TS[lvl]
            vz = np.where(vm.any((1, 2)))[0]
            if len(vz) >= 2:
                d[f"{lvl}_zc"] = float(vz.mean())
                d[f"{lvl}_h_cm"] = (vz[-1] - vz[0] + 1) * 0.3
                d[f"{lvl}_cut"] = int(vz[0] == 0 or vz[-1] == hu.shape[0] - 1)
                s = int(np.round(vz.mean()))
                d[f"{lvl}_vert_cm2"] = float(vm[s].sum() * 0.09)
                d[f"{lvl}_vert_ml"] = float(vm.sum() * VX)
                sl = hu[s]
                body = ndimage.binary_fill_holes(ndimage.binary_opening(sl > -500, iterations=2))
                lb, nn = ndimage.label(body)
                if nn:
                    body = lb == (np.bincount(lb.ravel())[1:].argmax() + 1)
                d[f"{lvl}_body_cm2"] = float(body.sum() * 0.09)
                ys = np.where(body.any(1))[0]; xs = np.where(body.any(0))[0]
                d[f"{lvl}_body_ap_cm"] = (ys[-1] - ys[0] + 1) * 0.3; d[f"{lvl}_body_w_cm"] = (xs[-1] - xs[0] + 1) * 0.3
                d[f"{lvl}_body_edge"] = int(body[:, 0].any() or body[:, -1].any() or body[0].any() or body[-1].any())
    except Exception as e:
        d["mask_err"] = str(e)[:40]
    return d


if __name__ == "__main__":
    L = [l.strip() for l in open(ids) if l.strip()]
    with Pool(npr) as p:
        res = [r for r in p.imap_unordered(feat, L, chunksize=8) if r is not None]
    pd.DataFrame(res).to_csv(out, index=False)
    print("done", len(res), "of", len(L), flush=True)
