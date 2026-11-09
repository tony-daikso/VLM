"""影像骨質疏鬆: trabecular-only ROI HU per vertebral level (T12, L1, L2, L3) from the 1.5 mm seg15 files
(Dataset292 vertebra masks + mid-sagittal HU slab hu_slab, 11 sagittal columns = 16.5 mm).

ROI (fixed before any label was looked at):
  1. V = level mask; < 800 voxels -> status no_mask.
  2. xc = median x of V; sagittal columns x in xc-3..xc+3 (+-4.5 mm, central part of the body) that lie inside the
     HU slab; < 3 such columns -> status off_slab.
  3. vertebral BODY footprint = most anterior 2D connected component of the mid-sagittal occupancy (xc+-2), same as
     geom15 (posterior elements are separated by the canal), holes filled.
  4. per column: S = V[:,:,x] AND body footprint (= whole body incl. cortex, used for cement / metal check);
     trabecular ROI = S eroded with a 9 x 7 rectangle (4 px = 6 mm along z -> away from both endplates,
     3 px = 4.5 mm along y -> away from anterior / posterior cortex). Lateral cortex is avoided by step 2.
  5. if ROI < 100 voxels -> erosion 3 x 2 px (4.5 / 3 mm) [erosion='small']; still < 50 -> status small_roi.
  6. ROI stats: n, median (= the measurement), mean, sd, p05, p95, frac > 400 / > 600 HU.
     BODY stats: n, n > 1000, n > 1500 (cement / metal), max.
  7. median outside [-300, 1200] -> status implausible.
Only train/val ids (split.csv read first) are processed; test files are never opened.
usage: python oh_roi.py SEG15_DIR SPLIT_CSV OUT_CSV [--jobs 4]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd

LEVELS = ["T12", "L1", "L2", "L3"]


def level_roi(lab, hu, x0, val):
    from scipy import ndimage as ndi
    r = dict(status="ok", erosion="main")
    V = lab == val
    if V.sum() < 800:
        return dict(status="no_mask")
    zz, yy, xx = np.nonzero(V)
    xc = int(round(np.median(xx)))
    cols = [x for x in range(xc - 3, xc + 4) if 0 <= x - x0 < hu.shape[2] and 0 <= x < V.shape[2]]
    if len(cols) < 3:
        return dict(status="off_slab")
    xs = slice(max(xc - 2, 0), xc + 3)
    occ = V[:, :, xs].mean(2)
    cc, k = ndi.label(occ > 0)
    comps = []
    for i in range(1, k + 1):
        m = cc == i; a = occ[m].sum()
        if a >= 20:
            comps.append((np.nonzero(m)[1].max(), a, i))
    if not comps:
        return dict(status="no_body")
    big = max(c[1] for c in comps)
    i_body = max([c for c in comps if c[1] >= 0.25 * big], key=lambda c: c[0])[2]
    foot = ndi.binary_fill_holes(cc == i_body)
    body_vals, roi_vals = [], []
    for ez, ey, tag in ((4, 3, "main"), (3, 2, "small")):
        roi_vals, body_vals = [], []
        st = np.ones((2 * ez + 1, 2 * ey + 1), bool)
        for x in cols:
            S = V[:, :, x] & foot
            h = hu[:, :, x - x0]
            body_vals.append(h[S])
            roi_vals.append(h[ndi.binary_erosion(S, st)])
        roi = np.concatenate(roi_vals); body = np.concatenate(body_vals)
        if len(roi) >= (100 if tag == "main" else 50):
            r["erosion"] = tag; break
    else:
        r["status"] = "small_roi"
    roi = roi.astype(np.float32); body = body.astype(np.float32)
    r.update(n_roi=len(roi), n_body=len(body), n_cols=len(cols),
             body_n1000=int((body > 1000).sum()), body_n1500=int((body > 1500).sum()),
             body_max=float(body.max()) if len(body) else np.nan)
    if len(roi):
        r.update(med=float(np.median(roi)), mean=float(roi.mean()), sd=float(roi.std()),
                 p05=float(np.percentile(roi, 5)), p95=float(np.percentile(roi, 95)),
                 f400=float((roi > 400).mean()), f600=float((roi > 600).mean()))
        if r["status"] == "ok" and not (-300 <= r["med"] <= 1200):
            r["status"] = "implausible"
    return r


def case(path):
    z = np.load(path, allow_pickle=True)
    if "hu_slab" not in z.files:
        return [dict(id=Path(path).stem, level=lv, status="no_hu") for lv in LEVELS]
    lab, labels = z["lab"], json.loads(str(z["labels"]))
    hu, x0 = z["hu_slab"], int(z["slab_x0"])
    rows = []
    for lv in LEVELS:
        key = f"vertebrae_{lv}"
        r = level_roi(lab, hu, x0, int(labels[key])) if key in labels else dict(status="no_mask")
        rows.append(dict(id=Path(path).stem, level=lv, **r))
    return rows


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("seg"); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    spl = pd.read_csv(a.split).set_index("study_id")["split"]
    ok = set(spl[spl.isin(["train", "val"])].index)
    paths = sorted(p for p in Path(a.seg).glob("*.npz") if p.stem in ok and ".tmp" not in p.name)
    print(len(paths), "train/val seg15 files", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=1, batch_size=16)(delayed(case)(p) for p in paths)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(a.out, index=False)
    print(df.groupby("level").status.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
