"""rib_ft prep (CT-rate CPU): raw Merlin CT -> canonical -> Exp L rib ROI (rib_boxes.npz, 3 mm rib bbox +-15 mm) ->
change_spacing 1.5 mm (identical to src/rib15_full.py, so the grid equals Exp L seg15) -> crop to the Dataset295 rib-label
bbox +-PAD voxels -> roi/{id}.npz: hu int16 (z,y,x), lab uint8 (Dataset295 labels; 1-24 ribs), sp (shape check).
usage: rib_prep.py IDS_TXT OUT_DIR [--jobs 12] ; ids in priority order; existing outputs skipped; touch OUT/STOP to stop."""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
CT_DIR = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
MARGIN, PAD = 15.0, 8
ap = argparse.ArgumentParser(); ap.add_argument("ids"); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--boxes", default=None); ap.add_argument("--seg", default=None)
ap.add_argument("--ct-dir", default=None); ap.add_argument("--watch", action="store_true", help="IDS = buffer dir; process + delete CTs as they arrive until IDS/DONE")
a = ap.parse_args(); OUT = Path(a.out); (OUT / "roi").mkdir(parents=True, exist_ok=True)
BOX = a.boxes or str(OUT / "rib_boxes.npz"); SEG = Path(a.seg or OUT / "seg15")


def rib_roi(canon, wc):
    vox = (np.linalg.inv(canon.affine) @ np.c_[wc, np.ones(len(wc))].T)[:3].T
    vlo, vhi = vox.min(0), vox.max(0); zooms = np.array(canon.header.get_zooms()[:3], float); sl = []
    for ax in range(3):
        a0 = int(np.floor(vlo[ax] - (MARGIN + 3) / zooms[ax])); a1 = int(np.ceil(vhi[ax] + (MARGIN + 3) / zooms[ax])) + 1
        sl.append(slice(max(a0, 0), min(a1, canon.shape[ax])))
    return canon.slicer[tuple(sl)]


def job(cid):
    import nibabel as nib
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    dst = OUT / "roi" / f"{cid}.npz"
    if dst.exists() or (OUT / "STOP").exists():
        return cid, "skip", 0
    t0 = time.time()
    try:
        sg = SEG / f"{cid}.npz"
        if not sg.exists():
            return cid, "noseg", 0
        z = np.load(sg); wc = BOXES.get(cid)
        if wc is None:
            return cid, "nobox", 0
        raw = nib.load(f"{a.ct_dir or CT_DIR}/{cid}.nii.gz")
        img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header); img.header.set_slope_inter(1, 0)
        canon = as_closest_canonical(img); roi = rib_roi(canon, wc)
        img15 = change_spacing(roi, [1.5, 1.5, 1.5], order=1, dtype=np.int32, nr_cpus=1)
        hu = np.asanyarray(img15.dataobj).astype(np.int16).transpose(2, 1, 0)
        if tuple(hu.shape) != tuple(z["shape15"]) or not np.allclose(img15.affine, z["affine"], atol=1e-3):
            return cid, f"gridmismatch {hu.shape} {tuple(z['shape15'])}", 0
        lab = np.zeros(hu.shape, np.uint8); off = z["off"]; L = z["lab"]
        lab[off[0]:off[0] + L.shape[0], off[1]:off[1] + L.shape[1], off[2]:off[2] + L.shape[2]] = L
        idx = np.argwhere((lab >= 1) & (lab <= 24))
        if len(idx) < 200:
            return cid, "fewribvox", 0
        lo = np.maximum(idx.min(0) - PAD, 0); hi = np.minimum(idx.max(0) + 1 + PAD, hu.shape)
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        tmp = OUT / "roi" / f"{cid}.tmp.npz"
        np.savez_compressed(tmp, hu=hu[sl], lab=lab[sl], lo=lo, shape15=np.array(hu.shape))
        tmp.rename(dst)
        return cid, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return cid, f"FAIL {e!r}"[:200], 0


zb = np.load(BOX); BOXES = {str(i): c for i, c in zip(zb["ids"], zb["corners"])}
def wjob(cid):
    r = job(cid)
    try: os.remove(f"{a.ct_dir}/{cid}.nii.gz")
    except OSError: pass
    return r


if __name__ == "__main__" and a.watch:
    from multiprocessing import Pool
    buf = a.ids; a.ct_dir = buf; busy = set()
    with Pool(a.jobs, maxtasksperchild=50) as p, open(OUT / "prep_status.tsv", "a") as fh:
        pend = []
        while True:
            for f in os.listdir(buf):
                if f.endswith(".nii.gz") and not f.startswith(".") and f[:-7] not in busy:
                    busy.add(f[:-7]); pend.append(p.apply_async(wjob, (f[:-7],)))
            for r in [r for r in pend if r.ready()]:
                cid, st, dt = r.get(); fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); pend.remove(r)
            if not pend and os.path.exists(f"{buf}/DONE") and not any(f.endswith(".nii.gz") for f in os.listdir(buf)): break
            if (OUT / "STOP").exists() and not pend: break
            time.sleep(1)
    print("WATCH_DONE", flush=True); sys.exit(0)

if __name__ == "__main__":
    from multiprocessing import Pool
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    todo = [i for i in ids if not (OUT / "roi" / f"{i}.npz").exists()]
    print(f"{len(todo)}/{len(ids)} to do, jobs {a.jobs}", flush=True)
    n = 0; t0 = time.time()
    with Pool(a.jobs, maxtasksperchild=50) as p, open(OUT / "prep_status.tsv", "a") as fh:
        for cid, st, dt in p.imap(job, todo, chunksize=1):
            fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s last {cid} {st} {dt:.1f}", flush=True)
    print("PREP_DONE", flush=True)
