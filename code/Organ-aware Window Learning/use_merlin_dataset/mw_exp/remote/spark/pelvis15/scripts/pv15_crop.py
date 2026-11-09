"""pelvis15 stage A (CPU): one TRUE 1.5 mm pelvis ROI per case, sampled straight from the raw Merlin CT (trilinear, one
HDD read per case; no 3 mm prep / upsampling). ROI = world-axis-aligned bbox of {both hip bones, sacrum, top 80 mm of
both femurs} from the TS Dataset297 3 mm 'total' mask + 12 mm margin (each axis capped at 384 mm around the bbox centre).
Labels (nearest from the 3 mm mask, then grown 2 voxels = 3 mm into background so the cortex is covered):
  1 hip_right 2 hip_left 3 sacrum 4 femur_right (top 80 mm) 5 femur_left (top 80 mm).
Out roi/{id}.npz: hu int16 (z,y,x along world S/A/R... array axes = world z,y,x), lab uint8, org (world x,y,z of voxel
[0,0,0]), xmid (midline world x = sacrum centroid x, else mean hip x), sp 1.5.
usage: pv15_crop.py ids.txt out_dir n_proc mask_dir [mask_dir ...]   ids = train/val only; done ids skipped; RAW15 env
overrides the raw-CT dir (relay buffer); --watch mode: ids.txt = buffer dir, crops + deletes arriving CTs until DONE."""
import sys, os, time, traceback, numpy as np, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
RAW = os.environ.get("RAW15", "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data")
SAC, FEML, FEMR, HIPL, HIPR = 25, 75, 76, 77, 78
CODE = {HIPR: 1, HIPL: 2, SAC: 3, FEMR: 4, FEML: 5}
LUT = np.zeros(256, np.uint8)
for _k, _v in CODE.items(): LUT[_k] = _v
SP, MARGIN, CAP, FEM_MM, GROW = 1.5, 12.0, 384.0, 80.0, 2


def world(aff, idx):
    return idx @ aff[:3, :3].T + aff[:3, 3]


def sample(arr, aff, pts, order, cval):
    """arr (i,j,k) sampled at world points pts [3,N] (x,y,z) through the inverse affine, local sub-block only."""
    ijk = (np.linalg.inv(aff) @ np.vstack([pts, np.ones(pts.shape[1])]))[:3]
    lo = np.floor(ijk.min(1)).astype(int) - 1; hi = np.ceil(ijk.max(1)).astype(int) + 2
    lo_c = np.maximum(lo, 0); hi_c = np.minimum(hi, arr.shape)
    if np.any(hi_c <= lo_c):
        return np.full(pts.shape[1], cval, np.float32)
    sub = np.asarray(arr[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]], np.float32)
    return ndimage.map_coordinates(sub, (ijk - lo_c[:, None]).astype(np.float32), order=order, mode="constant", cval=cval, prefilter=False)


def one(args):
    cid, out, mdirs = args
    dst = f"{out}/roi/{cid}.npz"
    if os.path.exists(dst):
        return cid, "skip", 0
    t0 = time.time()
    try:
        mp = next((f"{d}/{cid}.nii.gz" for d in mdirs if os.path.exists(f"{d}/{cid}.nii.gz")), None)
        if mp is None:
            return cid, "nomask", 0
        m = nib.load(mp); lab3 = np.asarray(m.dataobj).astype(np.uint8); maff = m.affine
        keep = np.isin(lab3, list(CODE))
        for fem in (FEMR, FEML):  # femur: top 80 mm only
            v = np.argwhere(lab3 == fem)
            if len(v):
                w = world(maff, v.astype(float)); drop = v[w[:, 2] < w[:, 2].max() - FEM_MM]
                keep[tuple(drop.T)] = False
        lab3 = np.where(keep, lab3, 0).astype(np.uint8)
        v = np.argwhere(keep)
        if len(v) < 500:
            return cid, f"fewvox {len(v)}", 0
        w = world(maff, v.astype(float)); lo = w.min(0) - MARGIN; hi = w.max(0) + MARGIN
        c = (lo + hi) / 2; half = np.minimum((hi - lo) / 2, CAP / 2); lo, hi = c - half, c + half
        n = np.maximum(np.round((hi - lo) / SP).astype(int), 8)            # (nx, ny, nz)
        xs, ys, zs = [lo[i] + SP * np.arange(n[i]) for i in range(3)]
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 0); shp = (n[2], n[1], n[0])
        L = sample(lab3, maff, pts, 0, 0).astype(np.uint8).reshape(shp)
        L = LUT[L]
        d, ind = ndimage.distance_transform_edt(L == 0, return_indices=True)
        grow = (L == 0) & (d <= GROW)
        L[grow] = L[tuple(i[grow] for i in ind)]
        del d, ind
        raw = nib.load(f"{RAW}/{cid}.nii.gz"); prox = raw.dataobj
        arr = np.asanyarray(prox.get_unscaled()); sl = prox.slope; it = prox.inter
        sl = 1.0 if sl is None or not np.isfinite(sl) else float(sl); it = 0.0 if it is None or not np.isfinite(it) else float(it)
        H = sample(arr, raw.affine, pts, 1, (-1024 - it) / sl) * sl + it
        H = np.clip(np.rint(H), -1024, 3071).astype(np.int16).reshape(shp)
        sac = np.argwhere(L == 3)
        xmid = float(xs[int(np.round(sac[:, 2].mean()))]) if len(sac) > 200 else float(np.mean(
            [xs[int(np.round(np.argwhere(L == k)[:, 2].mean()))] for k in (1, 2) if (L == k).any()] or [c[0]]))
        tmp = dst.replace(".npz", ".tmp.npz")
        np.savez_compressed(tmp, hu=H, lab=L, org=np.array([xs[0], ys[0], zs[0]], np.float32), xmid=np.float32(xmid), sp=np.float32(SP))
        os.rename(tmp, dst)
        cnt = np.bincount(L.ravel(), minlength=6)[1:]
        return cid, f"ok {shp} " + ",".join(map(str, cnt)), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return cid, "ERR " + repr(e)[:200] + " " + traceback.format_exc().splitlines()[-2][:120], time.time() - t0


def wjob(args):
    r = one(args)
    try: os.remove(f"{RAW}/{args[0]}.nii.gz")
    except OSError: pass
    return r


if __name__ == "__main__":
    ids, out, npr = sys.argv[1], sys.argv[2], int(sys.argv[3]); mdirs = sys.argv[4:]
    os.makedirs(f"{out}/roi", exist_ok=True)
    if os.path.isdir(ids):  # watch mode: relay buffer
        RAW = ids; busy = set(); pend = []
        with Pool(npr, maxtasksperchild=50) as pool, open(f"{out}/status_watch.tsv", "a") as fh:
            while True:
                for f in os.listdir(ids):
                    if f.endswith(".nii.gz") and not f.startswith(".") and f[:-7] not in busy:
                        busy.add(f[:-7]); pend.append(pool.apply_async(wjob, ((f[:-7], out, mdirs),)))
                for r in [r for r in pend if r.ready()]:
                    cid, st, dt = r.get(); fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); pend.remove(r)
                if not pend and (os.path.exists(f"{ids}/DONE") or os.path.exists(f"{out}/STOP")) and not any(
                        f.endswith(".nii.gz") for f in os.listdir(ids)):
                    break
                time.sleep(1)
        print("WATCH_DONE", flush=True); sys.exit(0)
    todo = [l.strip() for l in open(ids) if l.strip()]
    todo = [c for c in todo if not os.path.exists(f"{out}/roi/{c}.npz")]
    print(f"{len(todo)} to do", flush=True)
    n = 0; t0 = time.time()
    with Pool(npr, maxtasksperchild=50) as pool, open(f"{out}/status.tsv", "a") as fh:
        for cid, st, dt in pool.imap(one, [(c, out, mdirs) for c in todo], chunksize=1):
            fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            if os.path.exists(f"{out}/STOP"):
                print("STOP", flush=True); pool.terminate(); break
    print("CROP_DONE", flush=True)
