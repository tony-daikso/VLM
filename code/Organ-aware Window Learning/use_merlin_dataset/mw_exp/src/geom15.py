"""Experiment J: Genant-style vertebral geometry from the 1.5 mm Dataset292 vertebra masks (vert15_full.py seg15/).

Per level T8..L5 (T7 / sacrum used only as neighbours):
  1. mid-sagittal slab: x within the vertebra's median x +-2 voxels (7.5 mm); occupancy = mean of the binary mask
     over the slab (partial values -> sub-voxel), 2D map over (z = S/I, y = A/P), 1.5 mm pixels
  2. body = the most anterior 2D connected component of (occupancy > 0) (posterior elements are separated by the
     spinal canal in the mid-sagittal plane)
  3. tilt: body principal axes (occupancy-weighted PCA in mm); the axis of least variance = endplate normal; the
     body map is rotated (bilinear) so that the normal is vertical, upsampled x4 before rotation
  4. heights = column sums of the rotated occupancy (area / width -> sub-voxel), averaged over bands at 10-20 %
     (anterior), 45-55 % (middle), 80-90 % (posterior) of the AP depth; Ha/Hm/Hp in mm; depth in mm
  5. ratios: wedge Ha/Hp, biconcave Hm/max(Ha,Hp), min/max; neighbour reference = median of the max heights of up
     to 2 levels above and 2 below (present ones); loss_* = 1 - H_*/ref (Genant: 20-25 % grade 1, 25-40 % grade 2,
     > 40 % grade 3); crush = Hp / neighbours' Hp
  6. body HU: median HU of the body voxels in the mid-sagittal plane eroded by 2 px (hu_slab)
Case features (prefix g15_): min/argmin over levels of the ratios, max / 2nd max of the losses, counts of levels with
loss >= .20/.25/.40, n levels, min body HU, + the same after a per-level robust z-score (median/MAD of ALL train
cases, labels unused) -> max |z|.
Outputs: geom15_levels.csv (id, level, ...), geom15_features.csv (id, g15_*). Only train/val seg15 files exist.
usage: python geom15.py --seg-dirs DIR [DIR ...] --split split.csv --out OUTDIR [--jobs 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "sacrum"]
CORE = LEVELS[1:-1]
BANDS = {"a": (0.10, 0.20), "m": (0.45, 0.55), "p": (0.80, 0.90)}
UP = 4


def level_geom(lab, hu, slab_x0, val, zooms, zlim=None):
    from scipy import ndimage as ndi
    V = lab == val
    n = int(V.sum())
    if n < 800:
        return None
    zz, yy, xx = np.nonzero(V)
    if zlim is not None and (zz.min() + zlim[0] <= 0 or zz.max() + zlim[0] >= zlim[1] - 1):
        return None                                              # cut by the scan / ROI edge -> heights unreliable
    xc = int(round(np.median(xx)))
    xs = slice(max(xc - 2, 0), xc + 3)
    occ = V[:, :, xs].mean(2)                                    # (z, y) partial occupancy
    cc, k = ndi.label(occ > 0)
    if k == 0:
        return None
    comps = []
    for i in range(1, k + 1):
        m = cc == i
        a = occ[m].sum()
        if a < 20:
            continue
        comps.append((np.nonzero(m)[1].max(), a, i))           # anterior = larger y index (RAS: +y anterior)
    if not comps:
        return None
    big = max(c[1] for c in comps)
    i_body = max([c for c in comps if c[1] >= 0.25 * big], key=lambda c: c[0])[2]
    body = np.where(cc == i_body, occ, 0.0)
    z, y = np.nonzero(body)
    w = body[z, y]
    sz, sy = zooms[0], zooms[1]
    P = np.c_[z * sz, y * sy]
    mu = (P * w[:, None]).sum(0) / w.sum()
    C = np.cov((P - mu).T, aweights=w)
    ev, evec = np.linalg.eigh(C)
    nrm = evec[:, 0]                                             # least variance = endplate normal (z, y)
    if nrm[0] < 0:
        nrm = -nrm
    tilt = float(np.degrees(np.arctan2(nrm[1], nrm[0])))        # 0 = normal along z
    # isotropic upsampled map (mm/UP per pixel), then rotate so the normal is vertical
    z0, z1, y0, y1 = z.min(), z.max() + 1, y.min(), y.max() + 1
    pad = 6
    sub = np.pad(body[z0:z1, y0:y1], pad)
    up = ndi.zoom(sub, (sz * UP / 1.5, sy * UP / 1.5), order=1)
    rot = ndi.rotate(up, -tilt, reshape=True, order=1)
    px = 1.5 / UP                                                # mm per pixel
    colsum = rot.sum(0) * px                                     # height per column (mm)
    cols = np.nonzero(colsum > 0.1 * colsum.max())[0]
    if len(cols) < 4:
        return None
    # anterior = larger y index (array axis 1 increases anteriorly); rotation keeps the orientation
    yp_, ya_ = cols.min(), cols.max()
    depth = (ya_ - yp_ + 1) * px
    H = {}
    for b, (f0, f1) in BANDS.items():
        c0 = int(round(ya_ - f1 * (ya_ - yp_)))
        c1 = int(round(ya_ - f0 * (ya_ - yp_))) + 1
        H[b] = float(colsum[c0:c1].mean())
    # body HU in the mid-sagittal plane (hu_slab covers x in [slab_x0, slab_x0 + width))
    hu_med = np.nan
    if hu is not None:
        core = ndi.binary_erosion(cc == i_body, iterations=2)
        vals = [hu[:, :, x - slab_x0][core & V[:, :, x]] for x in range(xs.start, min(xs.stop, V.shape[2]))
                if 0 <= x - slab_x0 < hu.shape[2]]
        if vals:
            vals = np.concatenate(vals)
            vals = vals[vals > -1500]
            if len(vals) > 20:
                hu_med = float(np.median(vals))
    return {"Ha": H["a"], "Hm": H["m"], "Hp": H["p"], "depth": depth, "tilt": tilt,
            "vol_ml": n * zooms[0] * zooms[1] * zooms[2] / 1000, "body_area": float(body.sum() * sz * sy),
            "hu_body": hu_med}


def case_levels(path):
    z = np.load(path, allow_pickle=True)
    lab, labels = z["lab"], json.loads(str(z["labels"]))
    A = z["affine"]
    zooms_xyz = np.sqrt((A[:3, :3] ** 2).sum(0))
    zooms = zooms_xyz[::-1]                                      # (z, y, x)
    hu = z["hu_slab"] if "hu_slab" in z else None
    x0 = int(z["slab_x0"]) if "slab_x0" in z else 0
    zlim = (int(z["off"][0]), int(z["shape15"][0]))
    rows = []
    for lv in LEVELS:
        key = "sacrum" if lv == "sacrum" else f"vertebrae_{lv}"
        if key not in labels:
            continue
        g = level_geom(lab, hu, x0, int(labels[key]), zooms, zlim) if lv != "sacrum" else None
        if g is None:
            continue
        rows.append({"id": Path(path).stem, "level": lv, **g})
    return rows


def add_ratios(df):
    df = df.copy()
    df["Hmax"] = df[["Ha", "Hm", "Hp"]].max(axis=1)
    df["Hmin"] = df[["Ha", "Hm", "Hp"]].min(axis=1)
    df["wedge"] = df.Ha / df.Hp
    df["bicon"] = df.Hm / df[["Ha", "Hp"]].max(axis=1)
    df["minmax"] = df.Hmin / df.Hmax
    li = {l: i for i, l in enumerate(LEVELS)}
    df["li"] = df.level.map(li)
    out = []
    for cid, g in df.groupby("id", sort=False):
        g = g.sort_values("li")
        refs = {}
        for c in ("Ha", "Hm", "Hp", "Hmax", "vol_ml"):
            d = dict(zip(g.li, g[c]))
            refs[c] = [np.median([d[j] for j in (i - 2, i - 1, i + 1, i + 2) if j in d])
                       if any(j in d for j in (i - 2, i - 1, i + 1, i + 2)) else np.nan for i in g.li]
        g = g.assign(**{f"ref_{c}": v for c, v in refs.items()})
        out.append(g)
    df = pd.concat(out)
    for b in ("a", "m", "p"):
        df[f"loss_{b}"] = 1 - df[f"H{b}"] / df[f"ref_H{b}"]   # same-band neighbour reference (Genant-like)
    df["loss"] = df[["loss_a", "loss_m", "loss_p"]].max(axis=1)
    df["crush"] = df.Hp / df.ref_Hp
    df["loss_hmax"] = 1 - df.Hmin / df.ref_Hmax
    df["vol_ratio"] = df.vol_ml / df.ref_vol_ml
    return df[df.level.isin(CORE)]


def case_features(lv, train_ids):
    """per-level robust z (median / MAD over ALL train cases' levels; labels unused) + case summaries."""
    zc = ["wedge", "bicon", "minmax", "loss", "crush", "Hmin", "hu_body"]
    tr = lv[lv.id.isin(train_ids)]
    st = tr.groupby("level")[zc].agg(["median", lambda s: (s - s.median()).abs().median() * 1.4826 + 1e-6])
    for c in zc:
        med = lv.level.map(st[(c, "median")]); mad = lv.level.map(st[(c, "<lambda_0>")])
        lv[f"z_{c}"] = (lv[c] - med) / mad
    rows = {}
    for cid, g in lv.groupby("id", sort=False):
        loss = np.sort(g.loss.dropna().values)[::-1]
        r = {"g15_n_levels": len(g),
             "g15_min_wedge": g.wedge.min(), "g15_min_bicon": g.bicon.min(), "g15_min_minmax": g.minmax.min(),
             "g15_max_loss": loss[0] if len(loss) else np.nan, "g15_max2_loss": loss[1] if len(loss) > 1 else np.nan,
             "g15_max_loss_a": g.loss_a.max(), "g15_max_loss_m": g.loss_m.max(), "g15_max_loss_p": g.loss_p.max(),
             "g15_min_crush": g.crush.min(),
             "g15_n_loss20": int((g.loss >= .20).sum()), "g15_n_loss25": int((g.loss >= .25).sum()),
             "g15_n_loss40": int((g.loss >= .40).sum()),
             "g15_min_hu": g.hu_body.min(), "g15_med_hu": g.hu_body.median(),
             "g15_zmin_wedge": g.z_wedge.min(), "g15_zmin_bicon": g.z_bicon.min(), "g15_zmin_minmax": g.z_minmax.min(),
             "g15_zmax_loss": g.z_loss.max(), "g15_zmin_crush": g.z_crush.min(), "g15_zmin_Hmin": g.z_Hmin.min(),
             "g15_zmin_hu": g.z_hu_body.min(),
             "g15_max_loss_hmax": g.loss_hmax.max(), "g15_min_vol_ratio": g.vol_ratio.min(),
             "g15_lossmax_minus_med": (loss[0] - np.median(loss)) if len(loss) else np.nan}
        rows[cid] = r
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("id").reset_index()


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dirs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spl = pd.read_csv(a.split).set_index("study_id")["split"]
    spl = spl[spl.isin(["train", "val"])]
    files = {}
    for d in a.seg_dirs:
        for p in Path(d).glob("*.npz"):
            if p.stem in spl.index and ".tmp" not in p.name:
                files[p.stem] = p
    paths = sorted(files.values())
    if a.limit:
        paths = paths[:a.limit]
    cache = out / "geom15_levels_raw.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id"])
    todo = [p for p in paths if p.stem not in set(done.id)]
    print(f"{len(paths)} seg files, {len(todo)} new", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=2, batch_size=8)(delayed(case_levels)(p) for p in todo)
    new = pd.DataFrame([r for rr in res for r in rr])
    raw = pd.concat([done, new]) if len(done) else new
    raw = raw[raw.id.isin([p.stem for p in paths])]
    raw.to_csv(cache, index=False)
    lv = add_ratios(raw)
    lv.to_csv(out / "geom15_levels.csv", index=False)
    feat = case_features(lv, set(spl[spl == "train"].index))
    feat.to_csv(out / "geom15_features.csv", index=False)
    print(feat.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
