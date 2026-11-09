#!/usr/bin/env python3
"""S4: spleen volume and max abdominal-aortic diameter from the existing OWL TotalSegmentator (fast, 3 mm)
masks of the CG10k main series. Runs on CT-rate (CPU only, read-only on the masks); writes one small CSV.

  python s4_measure_ctrate.py --masks /media/daikso/CTDATA/cg10k/owl/main/masks --out measurements_ts.csv

Labels (TotalSegmentator 'total' v2): 1 spleen, 52 aorta, 32 vertebrae_T12, 31 vertebrae_L1.
- spleen_ml = voxel count x voxel volume; spleen_trunc = spleen touches the first/last axial slice (FOV cut,
  volume is a lower bound).
- aorta: abdominal part = axial slices from the T12 centroid (or L1 top if T12 absent) caudally to the last
  aortic slice (bifurcation). Per slice, the largest connected aortic component -> equivalent-circle diameter
  2*sqrt(A/pi) and the major-axis length from second moments (4*sqrt(lambda_max)). Max over slices is
  reported; the 3 mm grid gives roughly +/-3 mm error and an oblique (tortuous) aorta inflates the major axis.
"""
import argparse
import csv
import glob
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

SPLEEN, AORTA, T12, L1 = 1, 52, 32, 31


def one(path):
    sid = os.path.basename(path).replace(".nii.gz", "")
    out = {"study_id": sid}
    try:
        img = nib.as_closest_canonical(nib.load(path))  # RAS: axis2 increases superiorly
        a = np.asarray(img.dataobj).astype(np.uint8)
        dx, dy, dz = [float(v) for v in img.header.get_zooms()[:3]]
        vox_ml = dx * dy * dz / 1000.0
        sp = a == SPLEEN
        n = int(sp.sum())
        out["spleen_ml"] = round(n * vox_ml, 1) if n else ""
        if n:
            zs = np.where(sp.any(axis=(0, 1)))[0]
            out["spleen_trunc"] = int(zs.min() == 0 or zs.max() == a.shape[2] - 1)
        else:
            out["spleen_trunc"] = ""
        ao = a == AORTA
        zt = np.where((a == T12).any(axis=(0, 1)))[0]
        zl = np.where((a == L1).any(axis=(0, 1)))[0]
        if len(zt):
            ztop = int(round(zt.mean()))
        elif len(zl):
            ztop = int(zl.max())
        else:
            ztop = None
        zao = np.where(ao.any(axis=(0, 1)))[0]
        out.update(aorta_eqd_mm="", aorta_major_mm="", aorta_z_n="", aorta_reaches_bif="")
        if ztop is not None and len(zao):
            zlist = [z for z in zao if z <= ztop]
            best_e, best_m = 0.0, 0.0
            for z in zlist:
                lab, k = ndimage.label(ao[:, :, z])
                if k == 0:
                    continue
                sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
                comp = lab == (int(np.argmax(sizes)) + 1)
                area = comp.sum() * dx * dy
                eqd = 2 * np.sqrt(area / np.pi)
                ii, jj = np.nonzero(comp)
                if len(ii) >= 3:
                    c = np.cov(np.vstack([ii * dx, jj * dy]))
                    maj = 4 * np.sqrt(max(np.linalg.eigvalsh(c).max(), 0)) + dx  # + one voxel for discretisation
                else:
                    maj = eqd
                best_e, best_m = max(best_e, eqd), max(best_m, maj)
            if zlist:
                out.update(aorta_eqd_mm=round(best_e, 1), aorta_major_mm=round(best_m, 1), aorta_z_n=len(zlist),
                           aorta_reaches_bif=int(zao.min() > 0))
        out["ok"] = 1
    except Exception as e:  # noqa: BLE001
        out["ok"] = 0
        out["err"] = str(e)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.masks, "*.nii.gz")))
    with Pool(a.procs) as p:
        rows = p.map(one, paths, chunksize=16)
    cols = ["study_id", "spleen_ml", "spleen_trunc", "aorta_eqd_mm", "aorta_major_mm", "aorta_z_n",
            "aorta_reaches_bif", "ok", "err"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(len(rows), "done", sum(r["ok"] for r in rows), "ok")


if __name__ == "__main__":
    main()
