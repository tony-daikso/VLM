"""GS6 step 2 (Mac, CPU): raw CT (Passport) -> canonical RAS -> crop to the world box from gs6_boxes.py -> TS-identical
resampling to 1.5 mm (totalsegmentator change_spacing, order 1, int32) -> int16 NIfTI. Resumable (skips existing outputs).
The output is what TotalSegmentator full-res (Dataset291 part1 organs, 1.5 mm) sees after its own preprocessing.
usage: gs6_rawcrop.py boxes.csv raw_dir out_dir nproc"""
import sys, os, time, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
RAW, OUT = sys.argv[2], sys.argv[3]


def one(r):
    sid = r["study_id"]; fo = f"{OUT}/{sid}.nii.gz"
    if os.path.exists(fo): return sid, "skip", 0.0
    t = time.time()
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    raw = nib.load(f"{RAW}/{sid}.nii.gz")
    img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header); img.header.set_slope_inter(1, 0)
    img = as_closest_canonical(img)
    if r["status"] == "box":
        lo = np.array([r["wmin_x"], r["wmin_y"], r["wmin_z"]]); hi = np.array([r["wmax_x"], r["wmax_y"], r["wmax_z"]])
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        ijk = (np.linalg.inv(img.affine) @ np.c_[corners, np.ones(8)].T).T[:, :3]
        a = np.clip(np.floor(ijk.min(0)).astype(int), 0, None); b = np.minimum(np.ceil(ijk.max(0)).astype(int) + 1, img.shape[:3])
        if np.any(b - a < 8): return sid, "emptybox", time.time() - t
        img = img.slicer[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    img = change_spacing(img, [1.5, 1.5, 1.5], order=1, dtype=np.int32, nr_cpus=1)
    out = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.int16), img.affine)
    nib.save(out, fo + ".tmp.nii.gz"); os.replace(fo + ".tmp.nii.gz", fo)
    return sid, "ok", time.time() - t


def safe(r):
    try:
        return one(r)
    except Exception as e:  # noqa: BLE001
        return r["study_id"], "err " + repr(e)[:150], 0.0


if __name__ == "__main__":
    B = pd.read_csv(sys.argv[1]); B = B[B.status.isin(["box", "full"])]
    os.makedirs(OUT, exist_ok=True); npr = int(sys.argv[4])
    rows = B.to_dict("records"); t0 = time.time(); n = 0
    with Pool(npr) as p:
        for i, (sid, st, dt) in enumerate(p.imap_unordered(safe, rows, chunksize=1)):
            if st != "skip": n += 1
            if st.startswith("err") or st == "emptybox": print(sid, st, flush=True)
            if (i + 1) % 50 == 0: print(f"{i+1}/{len(rows)} last {dt:.1f}s elapsed {time.time()-t0:.0f}s", flush=True)
    print("RAWCROP_DONE", n, f"{time.time()-t0:.0f}s", flush=True)
