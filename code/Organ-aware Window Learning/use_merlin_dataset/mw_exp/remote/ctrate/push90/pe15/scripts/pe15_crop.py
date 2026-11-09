"""push90 task B (CT-rate CPU): 1.5 mm posterior-element crops per vertebra T8..L5 straight from the raw Merlin CT.
Reuses spine15_crop (seg15 = Dataset292 1.5 mm vertebra labels when >= 200 vox, else TS 3 mm mask) for the centres; the box
is wider / deeper than spine15 so it holds transverse processes, laminae, pars and the spinous process:
(z,y,x) = (24,72,80) at 1.5 mm = 36 x 108 x 120 mm along world S/A/R, centred at the vertebra centroid shifted 12 mm posterior.
Out crops/{id}.npz: hu int16 [10,24,72,80]; m packbits [10, 138240] (this level's vertebra mask); present bool[10];
src int8[10]; nvox int32[10]; centre f32[10,3] (world RAS mm, box centre).
usage: pe15_crop.py ids.txt out_dir n_proc seg15_dir mask_dir [mask_dir ...]   (train/val ids only; done ids skipped)"""
import sys, os, time, traceback, numpy as np, nibabel as nib
from multiprocessing import Pool
SCR = "/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/scripts"; sys.path.insert(0, SCR)
import frac15_crop as F
import spine15_crop as S15
SH = (24, 72, 80); G = F.grid(SH); SHIFT = np.array([0.0, -12.0, 0.0])


def one(args):
    cid, out, s15dir, mdirs = args
    dst = f"{out}/crops/{cid}.npz"
    if os.path.exists(dst): return cid, "skip", 0
    t0 = time.time()
    try:
        mp = next((f"{d}/{cid}.nii.gz" for d in mdirs if os.path.exists(f"{d}/{cid}.nii.gz")), None)
        s15 = f"{s15dir}/{cid}.npz"
        if mp is None and not os.path.exists(s15): return cid, "nomask", 0
        sp3 = [(None, 0)] * 10; lab3 = None
        if mp is not None:
            m = nib.load(mp); lab3 = np.asarray(m.dataobj).astype(np.uint8); maff = m.affine; sp3, _ = F.centres(lab3, maff)
        sp15 = [(None, 0)] * 10
        if os.path.exists(s15): labx, aff15, ids15, sp15 = S15.seg15_info(s15)
        raw = nib.load(f"{F.RAW}/{cid}.nii.gz"); prox = raw.dataobj
        arr = np.asanyarray(prox.get_unscaled()); sl = prox.slope; it = prox.inter
        sl = 1.0 if sl is None or not np.isfinite(sl) else float(sl); it = 0.0 if it is None or not np.isfinite(it) else float(it)
        raff = raw.affine; U = 10; n = int(np.prod(SH))
        H = np.full((U,) + SH, -1024, np.int16); M = np.zeros((U, n), bool)
        pres = np.zeros(U, bool); src = np.full(U, -1, np.int8); nv = np.zeros(U, np.int32); C = np.full((U, 3), np.nan, np.float32)
        for k in range(U):
            if sp15[k][0] is not None: c, nv[k], src[k] = sp15[k][0], sp15[k][1], 1
            elif sp3[k][0] is not None: c, nv[k], src[k] = sp3[k][0], sp3[k][1], 0
            else: continue
            c = np.asarray(c, float) + SHIFT
            v = F.sample(arr, raff, c, G, SH, 1, (-1024 - it) / sl) * sl + it
            H[k] = np.clip(np.rint(v), -1024, 3071).astype(np.int16)
            if src[k] == 1: M[k] = (F.sample(labx, aff15, c, G, SH, 0, 0).astype(np.uint8) == ids15[k]).ravel()
            else: M[k] = (F.sample(lab3, maff, c, G, SH, 0, 0).astype(np.uint8) == F.SP_LV[k]).ravel()
            pres[k] = True; C[k] = c
        tmp = dst.replace(".npz", ".tmp.npz")
        np.savez_compressed(tmp, hu=H, m=np.packbits(M, 1), present=pres, src=src, nvox=nv, centre=C)
        os.rename(tmp, dst)
        return cid, f"ok n{int(pres.sum())} s15_{int((src == 1).sum())}", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return cid, "ERR " + repr(e)[:200] + " " + traceback.format_exc().splitlines()[-2][:120], time.time() - t0


if __name__ == "__main__":
    ids, out, npr, s15dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; mdirs = sys.argv[5:]
    os.makedirs(f"{out}/crops", exist_ok=True)
    todo = [l.strip() for l in open(ids) if l.strip()]
    todo = [c for c in todo if not os.path.exists(f"{out}/crops/{c}.npz")]
    print(f"{len(todo)} to do", flush=True); n = 0; nerr = 0; t0 = time.time()
    with Pool(npr, maxtasksperchild=50) as pool, open(f"{out}/status.tsv", "a") as fh:
        for cid, st, dt in pool.imap(one, [(c, out, s15dir, mdirs) for c in todo], chunksize=1):
            fh.write(f"{cid}\t{st}\t{dt:.1f}\n"); fh.flush(); n += 1
            nerr = nerr + 1 if st.startswith("ERR") else 0
            if nerr >= 30: print("30 consecutive errors -> abort", flush=True); pool.terminate(); break
            if n % 200 == 0: print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            if os.path.exists(f"{out}/STOP"): print("STOP", flush=True); pool.terminate(); break
    print("CROP_DONE", flush=True)
