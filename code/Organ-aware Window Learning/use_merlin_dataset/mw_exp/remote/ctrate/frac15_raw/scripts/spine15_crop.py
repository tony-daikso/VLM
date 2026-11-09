"""spine15_ft stage A (CPU, CT-rate): TRUE 1.5 mm per-vertebra crops (T8..L5) from the raw Merlin CT (one HDD read/case).
Same unit layout / FOV / orientation as spine_ft (10 units, 40x48x40 at 1.5 mm = 60x72x60 mm, array (z,y,x) along world
S/A/R), but HU are sampled with trilinear interpolation straight from the raw CT (no 3 mm prep, no x2 upsampling).
Centres + masks: Exp J seg15 (Dataset292 1.5 mm vertebra labels; vert15_seg15/{id}.npz) when that vertebra is present
there (>= 200 voxels at 1.5 mm), else the TS Dataset297 3 mm mask (frac15_crop.centres, >= 30 voxels at 3 mm).
The 3 mm-mask version of every unit mask is stored as well (m3) for ablation / fallback.
Out crops/{id}.npz: hu int16 [10,40,48,40]; m, m3 packbits [10, 76800]; present bool[10]; src int8[10] (1 seg15, 0 3 mm,
-1 absent); nvox int32[10]; centre f32[10,3] (world RAS mm).
usage: spine15_crop.py ids.txt out_dir n_proc seg15_dir mask_dir [mask_dir ...]   (ids = train/val only; done ids skipped)"""
import sys, os, json, time, traceback, numpy as np, nibabel as nib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frac15_crop as F  # RAW, SP_LV (3 mm ids T8..L5), grid/sample/world/centres
F.RAW = os.environ.get("RAW15", F.RAW)  # buffer dir for CTs relayed from the Mac Passport

SH = (40, 48, 40)
G = F.grid(SH)
LV_NAMES = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]  # same order as F.SP_LV
MIN15 = 200


def seg15_info(path):
    z = np.load(path)
    names = json.loads(str(z["labels"]))
    ids = [names.get(f"vertebrae_{n}") for n in LV_NAMES]
    lab = z["lab"]  # (z,y,x) crop of the 1.5 mm grid at offset off (z,y,x)
    off = z["off"].astype(float)
    T = np.eye(4); T[:3, 3] = off[::-1]
    aff = z["affine"] @ T  # (x,y,z) index of the crop -> world
    labx = np.ascontiguousarray(lab.transpose(2, 1, 0))
    cs = []
    for l in ids:
        v = np.argwhere(labx == l) if l is not None else np.zeros((0, 3))
        cs.append((F.world(aff, v.mean(0)[None])[0], len(v)) if len(v) >= MIN15 else (None, len(v)))
    return labx, aff, ids, cs


def one(args):
    cid, out, s15dir, mdirs = args
    dst = f"{out}/crops/{cid}.npz"
    if os.path.exists(dst):
        return cid, "skip", 0
    t0 = time.time()
    try:
        mp = next((f"{d}/{cid}.nii.gz" for d in mdirs if os.path.exists(f"{d}/{cid}.nii.gz")), None)
        s15 = f"{s15dir}/{cid}.npz"
        if mp is None and not os.path.exists(s15):
            return cid, "nomask", 0
        sp3 = [(None, 0)] * 10; lab3 = None
        if mp is not None:
            m = nib.load(mp); lab3 = np.asarray(m.dataobj).astype(np.uint8); maff = m.affine
            sp3 = []  # spine part of F.centres only (skips the pelvis distance transforms)
            for l in F.SP_LV:
                v = np.argwhere(lab3 == l)
                sp3.append((F.world(maff, v.mean(0)[None])[0], len(v)) if len(v) >= 30 else (None, len(v)))
        sp15 = [(None, 0)] * 10; labx = None
        if os.path.exists(s15):
            labx, aff15, ids15, sp15 = seg15_info(s15)
        raw = nib.load(f"{F.RAW}/{cid}.nii.gz"); prox = raw.dataobj
        arr = np.asanyarray(prox.get_unscaled()); sl = prox.slope; it = prox.inter
        sl = 1.0 if sl is None or not np.isfinite(sl) else float(sl); it = 0.0 if it is None or not np.isfinite(it) else float(it)
        raff = raw.affine
        U = 10; n = int(np.prod(SH))
        H = np.full((U,) + SH, -1024, np.int16); M = np.zeros((U, n), bool); M3 = np.zeros((U, n), bool)
        pres = np.zeros(U, bool); src = np.full(U, -1, np.int8); nv = np.zeros(U, np.int32); C = np.full((U, 3), np.nan, np.float32)
        for k in range(U):
            if sp15[k][0] is not None:
                c, nv[k], src[k] = sp15[k][0], sp15[k][1], 1
            elif sp3[k][0] is not None:
                c, nv[k], src[k] = sp3[k][0], sp3[k][1], 0
            else:
                continue
            v = F.sample(arr, raff, c, G, SH, 1, (-1024 - it) / sl) * sl + it
            H[k] = np.clip(np.rint(v), -1024, 3071).astype(np.int16)
            if lab3 is not None:
                M3[k] = (F.sample(lab3, maff, c, G, SH, 0, 0).astype(np.uint8) == F.SP_LV[k]).ravel()
            if src[k] == 1:
                M[k] = (F.sample(labx, aff15, c, G, SH, 0, 0).astype(np.uint8) == ids15[k]).ravel()
            else:
                M[k] = M3[k]
            pres[k] = True; C[k] = c
        tmp = dst.replace(".npz", ".tmp.npz")
        np.savez_compressed(tmp, hu=H, m=np.packbits(M, 1), m3=np.packbits(M3, 1), present=pres, src=src, nvox=nv, centre=C)
        os.rename(tmp, dst)
        return cid, f"ok n{int(pres.sum())} s15_{int((src == 1).sum())}", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return cid, "ERR " + repr(e)[:200] + " " + traceback.format_exc().splitlines()[-2][:120], time.time() - t0


if __name__ == "__main__":
    ids, out, npr, s15dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; mdirs = sys.argv[5:]
    os.makedirs(f"{out}/crops", exist_ok=True)
    todo = [l.strip() for l in open(ids) if l.strip()]
    todo = [c for c in todo if not os.path.exists(f"{out}/crops/{c}.npz")]
    print(f"{len(todo)} to do", flush=True)
    n = 0; t0 = time.time()
    with Pool(npr, maxtasksperchild=50) as pool, open(f"{out}/status.tsv", "a") as fh:
        for cid, st, dt in pool.imap(one, [(c, out, s15dir, mdirs) for c in todo], chunksize=1):
            fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            if os.path.exists(f"{out}/STOP"):
                print("STOP", flush=True); pool.terminate(); break
    print("CROP_DONE", flush=True)
