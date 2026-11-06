"""Experiment K (rib high-res path), step 1: straightened ("unfolded") ribs from the raw CT. CPU only.

Per case:
  raw CT (ssh cat from the CT-rate HDD, or a local file; native ~0.65 mm in-plane, ~1 mm slices) +
  existing TotalSegmentator Dataset297 3 mm mask (outputs/masks/{id}.nii.gz, canonical RAS grid; world coords match CT).
  For ribs 5-12 per side:
    1. centerline: largest component of the (closed) 3 mm rib label -> geodesic distance from one end (Dijkstra on the
       26-neighbour voxel graph) -> centroids of 3 mm geodesic bins -> smoothed -> resampled every 1 mm of arc length.
    2. HU refinement: at each point, the centroid of bone (HU >= 200) in the cross-section disc (r <= 9 mm) pulls the
       point (2 passes, re-smoothed) -> centerline at ~1 mm accuracy without a high-res segmentation.
    3. curved planar reformation: cross-section planes (normal = tangent; in-plane axes = superior axis projected,
       and tangent x that) sampled 32 x 32 at 1 mm, trilinear -> (L, 32, 32). Index 0 = posterior/medial end.
  HU stored as uint8: q = clip((HU + 1000) / 12, 0, 255)  (range -1000..2060 HU, 12 HU steps).
Outputs (out dir):
  cpr/{id}.npz : vol_<rib> uint8 (L,32,32), ctr_<rib> float32 (L,3) world RAS mm, meta json (per rib: length_mm,
                 n_vox3, trunc (touches the scan z-edge), comp_frac (largest component share)), spacing info
  claims/{id}, timing_w{wid}.csv, failed.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

CT_HOST = "daikso@192.168.0.189"
CT_DIR = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
RIB_LAB = {f"L{k}": 91 + k for k in range(5, 13)} | {f"R{k}": 103 + k for k in range(5, 13)}  # Dataset297 ids
PS = 32       # cross-section size (voxels)
PSP = 1.0     # cross-section spacing (mm)
STEP = 1.0    # arc step (mm)
QS, QO = 12.0, -1000.0


AZ_BASE = "https://aimistanforddatasets01.blob.core.windows.net/merlinabdominalctdataset/merlin_data"


def fetch_azure(cid, dst: Path, sas_file):
    """Download from the AIMI Azure container (SAS read from a 600 file, never printed)."""
    sas = Path(sas_file).expanduser().read_text().strip()
    tmp = dst.with_suffix(".part")
    for att in range(4):
        r = subprocess.run(["curl", "-sf", "--max-time", "2400", "--speed-limit", "30000", "--speed-time", "120",
                            "-o", str(tmp), f"{AZ_BASE}/{cid}.nii.gz?{sas}"], timeout=2500)
        if r.returncode == 0:
            tmp.rename(dst)
            return dst
        time.sleep(10 * (att + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"azure curl failed rc={r.returncode}")


def fetch(cid, dst: Path, local_dir):
    if local_dir:
        p = Path(local_dir) / f"{cid}.nii.gz"
        if p.exists():
            return p
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["ssh", "-c", "aes128-gcm@openssh.com", "-o", "BatchMode=yes", CT_HOST,
                        f"cat {CT_DIR}/{cid}.nii.gz"], stdout=fh, check=True, timeout=1200)
    tmp.rename(dst)
    return dst


def smooth_curve(P, sigma):
    from scipy.ndimage import gaussian_filter1d
    if len(P) < 4:
        return P
    return gaussian_filter1d(P, sigma, axis=0, mode="nearest")


def resample_arc(P, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < step * 2:
        return P[:1], d[-1]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, P[:, k]) for k in range(3)], 1), d[-1]


def centerline_from_mask(vox, A3):
    """vox: (N,3) voxel indices of one rib on the 3 mm grid. Returns ordered world points (M,3) and the component share."""
    from scipy import ndimage as ndi
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    lo = vox.min(0) - 3
    shp = vox.max(0) - lo + 4
    m = np.zeros(shp, bool)
    v = vox - lo
    m[tuple(v.T)] = True
    m = ndi.binary_closing(m, structure=ndi.generate_binary_structure(3, 3), iterations=1) | m
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None, 0.0
    sz = np.bincount(lab.ravel())[1:]
    big = int(np.argmax(sz)) + 1
    frac = float(sz[big - 1] / sz.sum())
    pts = np.argwhere(lab == big)
    if len(pts) < 8:
        return None, frac
    idx = -np.ones(shp, np.int64)
    idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols, ws = [], [], []
    for off in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
        if not off.any() or tuple(off) < (0, 0, 0):
            continue
        q = pts + off
        ok = (q >= 0).all(1) & (q < shp).all(1)
        j = np.full(len(pts), -1)
        j[ok] = idx[tuple(q[ok].T)]
        k = j >= 0
        rows.append(np.nonzero(k)[0]); cols.append(j[k]); ws.append(np.full(k.sum(), np.linalg.norm(off)))
    r, c, w = map(np.concatenate, (rows, cols, ws))
    G = coo_matrix((w, (r, c)), shape=(len(pts), len(pts))).tocsr()
    d0 = dijkstra(G, directed=False, indices=0)
    e1 = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    d1 = dijkstra(G, directed=False, indices=e1)
    d1 = np.where(np.isfinite(d1), d1, np.nan)
    world = (A3[:3, :3] @ (pts + lo).T).T + A3[:3, 3]
    b = np.floor(d1 / 1.0).astype(float)  # bins of 1 voxel (3 mm) of geodesic length
    ok = np.isfinite(b)
    ub = np.unique(b[ok])
    C = np.stack([world[ok][b[ok] == u].mean(0) for u in ub])
    return C, frac


def frames(P):
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    S = np.array([0.0, 0.0, 1.0])
    N1 = S - (T @ S)[:, None] * T
    nn = np.linalg.norm(N1, axis=1)
    bad = nn < 0.2
    if bad.any():  # tangent ~ superior: use the anterior axis instead
        Aax = np.array([0.0, 1.0, 0.0])
        N1[bad] = Aax - (T[bad] @ Aax)[:, None] * T[bad]
        nn = np.linalg.norm(N1, axis=1)
    N1 /= nn[:, None]
    N2 = np.cross(T, N1)
    return T, N1, N2


def sample_planes(ct, Ainv, P, N1, N2, ps=PS, sp=PSP):
    from scipy.ndimage import map_coordinates
    g = (np.arange(ps) - (ps - 1) / 2) * sp
    U, V = np.meshgrid(g, g, indexing="ij")
    W = P[:, None, None, :] + U[None, :, :, None] * N1[:, None, None, :] + V[None, :, :, None] * N2[:, None, None, :]
    W = W.reshape(-1, 3)
    vx = (Ainv[:3, :3] @ W.T) + Ainv[:3, 3:4]
    out = map_coordinates(ct, vx, order=1, mode="constant", cval=-1024.0, prefilter=False)
    return out.reshape(len(P), ps, ps)


def refine(ct, Ainv, P, passes=2, r=9.0, thr=200.0):
    for _ in range(passes):
        T, N1, N2 = frames(P)
        hu = sample_planes(ct, Ainv, P, N1, N2, ps=19, sp=1.0)
        g = np.arange(19) - 9.0
        U, V = np.meshgrid(g, g, indexing="ij")
        disc = (U ** 2 + V ** 2) <= r * r
        w = ((hu >= thr) & disc[None]).astype(float)
        sw = w.sum((1, 2))
        du = np.where(sw > 3, (w * U).sum((1, 2)) / np.maximum(sw, 1), 0)
        dv = np.where(sw > 3, (w * V).sum((1, 2)) / np.maximum(sw, 1), 0)
        shift = du[:, None] * N1 + dv[:, None] * N2
        shift = smooth_curve(shift, 4.0)
        P = smooth_curve(P + shift, 2.0)
        P, _ = resample_arc(P, STEP)
    return P


def process(ct, Ac, m3, A3):
    Ainv = np.linalg.inv(Ac)
    zmax = m3.shape[2] - 1
    vols, ctrs, meta = {}, {}, {}
    # midline x (world) from the spine: median x of all rib voxels' mirror is unreliable; use the mean of left+right ribs
    for name, lab in RIB_LAB.items():
        vox = np.argwhere(m3 == lab)
        if len(vox) < 15:
            continue
        C, frac = centerline_from_mask(vox, A3)
        if C is None or len(C) < 4:
            continue
        P = smooth_curve(C, 1.0)
        P, L0 = resample_arc(P, STEP)
        if len(P) < 15:
            continue
        P = refine(ct, Ainv, P)
        if len(P) < 15:
            continue
        # posterior end first (RAS: smaller y = posterior)
        if P[0, 1] > P[-1, 1]:
            P = P[::-1].copy()
        T, N1, N2 = frames(P)
        if name.startswith("R"):  # mirror the in-plane axis so left and right ribs look alike (lateral = +v)
            N2 = -N2
        hu = sample_planes(ct, Ainv, P, N1, N2)
        q = np.clip(np.round((hu - QO) / QS), 0, 255).astype(np.uint8)
        vols[name] = q
        ctrs[name] = P.astype(np.float32)
        meta[name] = dict(length_mm=float(len(P) * STEP), n_vox3=int(len(vox)), comp_frac=round(frac, 3),
                          trunc=bool((vox[:, 2] <= 0).any() or (vox[:, 2] >= zmax).any()))
    return vols, ctrs, meta


def claim(out: Path, cid: str) -> bool:
    try:
        os.close(os.open(out / "claims" / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    import nibabel as nib
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-dirs", nargs="+", required=True)
    ap.add_argument("--ct-local", default=None)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-ct", action="store_true")
    ap.add_argument("--azure-par", type=int, default=0, help="download from Azure with this many parallel streams")
    ap.add_argument("--sas-file", default="~/OWL/.azure_sas")
    ap.add_argument("--reverse", action="store_true", help="walk the id list from the end of each tier")
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("cpr", "claims", "ct"):
        (out / d).mkdir(parents=True, exist_ok=True)
    ids = [l.strip().split(",")[0] for l in open(a.ids) if l.strip() and not l.startswith("study_id")]
    if a.reverse:
        ids = ids[::-1]
    mdirs = [Path(d) for d in a.mask_dirs]
    logf = out / f"timing_w{a.wid}.csv"
    new = not logf.exists()
    fh = open(logf, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["id", "pull", "load", "proc", "save", "n_ribs", "status"])

    def gen():
        for cid in ids:
            if (out / "cpr" / f"{cid}.npz").exists():
                continue
            if not any((d / f"{cid}.nii.gz").exists() for d in mdirs):
                continue
            if claim(out, cid):
                yield cid

    def handle(cid, ctp, T):
        st, nr = "ok", 0
        try:
            if isinstance(ctp, Exception) or ctp is None:
                raise RuntimeError(f"pull failed: {ctp!r}")
            t = time.time()
            raw = nib.load(str(ctp))
            ct = np.asanyarray(raw.dataobj).astype(np.float32)
            mp = next(d / f"{cid}.nii.gz" for d in mdirs if (d / f"{cid}.nii.gz").exists())
            mi = nib.load(str(mp))
            m3 = np.asanyarray(mi.dataobj).astype(np.uint8)
            T["load"] = time.time() - t; t = time.time()
            vols, ctrs, meta = process(ct, raw.affine, m3, mi.affine)
            T["proc"] = time.time() - t; t = time.time()
            nr = len(vols)
            arrs = {f"vol_{k}": v for k, v in vols.items()} | {f"ctr_{k}": v for k, v in ctrs.items()}
            tmp = out / "cpr" / f"{cid}.tmp.npz"
            np.savez_compressed(tmp, meta=json.dumps(meta), zooms=np.array(raw.header.get_zooms()[:3]), **arrs)
            tmp.rename(out / "cpr" / f"{cid}.npz")
            T["save"] = time.time() - t
        except Exception as e:  # noqa: BLE001
            st = f"fail:{e!r}"[:300]
            with open(out / "failed.txt", "a") as f:
                f.write(f"{cid}\t{st}\n")
        finally:
            if not a.ct_local and not a.keep_ct and isinstance(ctp, Path) and ctp.exists():
                ctp.unlink()
        wr.writerow([cid, *(round(T.get(k, np.nan), 2) for k in ("pull", "load", "proc", "save")), nr, st]); fh.flush()
        print(f"w{a.wid} {cid} {st[:80]} ribs={nr} " + " ".join(f"{k}={v:.1f}" for k, v in T.items()), flush=True)

    g = gen()
    pre = {}

    def prefetch(cid):
        t = time.time()
        try:
            pre[cid] = fetch(cid, out / "ct" / f"{cid}.nii.gz", a.ct_local)
        except Exception as e:  # noqa: BLE001
            pre[cid] = e
        pre[cid + "_t"] = time.time() - t

    def start():
        cid = next(g, None)
        if cid is None:
            return None, None
        th = threading.Thread(target=prefetch, args=(cid,), daemon=True)
        th.start()
        return cid, th

    if a.azure_par:
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
        pool = ThreadPoolExecutor(a.azure_par)

        def dl(cid):
            t = time.time()
            try:
                return cid, fetch_azure(cid, out / "ct" / f"{cid}.nii.gz", a.sas_file), time.time() - t
            except Exception as e:  # noqa: BLE001
                return cid, e, time.time() - t
        fl = set()
        for _ in range(a.azure_par):
            c = next(g, None)
            if c is not None:
                fl.add(pool.submit(dl, c))
        n = 0
        while fl:
            done, fl = wait(fl, return_when=FIRST_COMPLETED)
            for f in done:
                c, ctp, tp = f.result()
                handle(c, ctp, {"pull": tp})
                n += 1
                nx = next(g, None) if not (a.limit and n >= a.limit) else None
                if nx is not None:
                    fl.add(pool.submit(dl, nx))
        return

    n = 0
    cid, th = start()
    while cid is not None:
        th.join()
        last = a.limit and n + 1 >= a.limit
        ncid, nth = (None, None) if last else start()
        T = {"pull": pre.pop(cid + "_t", np.nan)}
        ctp = pre.pop(cid, None)
        handle(cid, ctp, T)
        n += 1
        cid, th = ncid, nth


if __name__ == "__main__":
    main()
