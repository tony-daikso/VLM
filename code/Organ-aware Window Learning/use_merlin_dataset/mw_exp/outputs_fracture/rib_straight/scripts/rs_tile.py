"""rib_straight: straightened (curved-planar) rib tiles from the 1.5 mm rib ROIs (rib_ft/roi/{id}.npz: hu int16 (z,y,x),
lab uint8 Dataset295: 1-12 rib_left_1..12, 13-24 rib_right_1..12; RAS canonical, 1.5 mm isotropic).
Per rib k with >= --minvox voxels:
  1. voxel graph (neighbours within 1.75 vox); components >= 15 vox; greedily bridge the nearest component while the gap
     <= --bridge vox (fragmented / fractured ribs stay one rib); far blobs dropped.
  2. geodesic double sweep -> end e0 -> geodesic arc parameter s; centroid per 2-vox bin of s; gaps interpolated;
     Gaussian smoothing; resample at 1 vox (1.5 mm) arc steps.
  3. orient posterior (smaller y) end first; extend both ends straight by --margin vox (real context), then extend to
     >= --minlen and to a multiple of 16.
  4. frame: T = tangent, u = superior axis projected off T (fallback: previous u), v = T x u; sample a --cs x --cs
     cross-section (1.5 mm grid) at every arc step -> dense tile [L, cs, cs] (trilinear, outside-ROI = nearest edge).
Output tiles/{id}.npz: hu int16 [sumL,cs,cs] (ribs concatenated along arc), m uint8 packbits of the rib-k mask along the
last axis, tab int32 [R,6] = (rib k, start, L, m0 (first true-rib row), n (true rib rows), nvox), oor float32 [R]
(fraction of samples outside the ROI), cl float16 [sumL,3] centerline in shape15 (Exp L 1.5 mm grid) voxel coords.
usage: rs_tile.py ROI OUT --ids order.txt [--jobs 14] [--part i/n]; skip existing; touch OUT/STOP to stop."""
import os, sys, time, argparse, zlib, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("roi"); ap.add_argument("out"); ap.add_argument("--ids", default="")
ap.add_argument("--jobs", type=int, default=14); ap.add_argument("--part", default="0/1"); ap.add_argument("--cs", type=int, default=32)
ap.add_argument("--minvox", type=int, default=60); ap.add_argument("--bridge", type=float, default=20.0)
ap.add_argument("--margin", type=int, default=8); ap.add_argument("--minlen", type=int, default=64)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--qc", default="", help="write a PNG montage of this id and exit")
a = ap.parse_args(); TD = os.path.join(a.out, "tiles"); os.makedirs(TD, exist_ok=True)


def centerline(P):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.ndimage import gaussian_filter1d
    N = len(P); pr = cKDTree(P).query_pairs(1.75, output_type="ndarray")
    if len(pr) == 0: return None
    w = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1)
    nc, cl = connected_components(coo_matrix((w, (pr[:, 0], pr[:, 1])), shape=(N, N)), directed=False)
    sz = np.bincount(cl, minlength=nc); order = [c for c in np.argsort(-sz) if sz[c] >= 15]
    if not order: return None
    keep = [order[0]]; rest = order[1:]; br = []
    while rest:
        cur = np.where(np.isin(cl, keep))[0]; t = cKDTree(P[cur]); best = None
        for c in rest:
            ii = np.where(cl == c)[0]; d, j = t.query(P[ii]); q = int(np.argmin(d))
            if best is None or d[q] < best[0]: best = (d[q], c, ii[q], cur[j[q]])
        if best[0] > a.bridge: break
        keep.append(best[1]); rest.remove(best[1]); br.append((best[2], best[3], best[0]))
    idx = np.where(np.isin(cl, keep))[0]; re = -np.ones(N, int); re[idx] = np.arange(len(idx))
    e = pr[(re[pr[:, 0]] >= 0) & (re[pr[:, 1]] >= 0)]; we = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    if br:
        b = np.array([(x, y) for x, y, _ in br]); e = np.r_[e, b]; we = np.r_[we, [g for _, _, g in br]]
    n = len(idx); G = coo_matrix((we, (re[e[:, 0]], re[e[:, 1]])), shape=(n, n)).tocsr(); Q = P[idx]
    d0 = dijkstra(G, directed=False, indices=0); e0 = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    s = dijkstra(G, directed=False, indices=e0); s = np.where(np.isfinite(s), s, np.nan)
    ok = np.isfinite(s); Q, s = Q[ok], s[ok]
    nb = int(s.max() // 2) + 1
    if nb < 3: return None
    bi = (s // 2).astype(int); cnt = np.bincount(bi, minlength=nb)
    C = np.stack([np.bincount(bi, weights=Q[:, d], minlength=nb) for d in range(3)], 1)
    has = cnt > 0; C[has] /= cnt[has, None]; xs = np.arange(nb)
    for d in range(3): C[~has, d] = np.interp(xs[~has], xs[has], C[has, d])
    C = gaussian_filter1d(C, 1.5, axis=0, mode="nearest")
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    L = int(np.floor(seg[-1])) + 1
    if L < 4: return None
    t = np.arange(L, dtype=float); C = np.stack([np.interp(t, seg, C[:, d]) for d in range(3)], 1)
    if C[0, 1] > C[-1, 1]: C = C[::-1].copy()   # posterior (smaller y) end first
    return C, len(idx), len(br)


def extend(C):
    n = len(C); k = min(5, n - 1)
    t0 = C[0] - C[k]; t0 /= max(np.linalg.norm(t0), 1e-6); t1 = C[-1] - C[-1 - k]; t1 /= max(np.linalg.norm(t1), 1e-6)
    f = b = a.margin; tot = n + 2 * a.margin
    if tot < a.minlen: ex = a.minlen - tot; f += ex // 2; b += ex - ex // 2; tot = a.minlen
    b += (-tot) % 16
    pre = C[0] + t0 * np.arange(f, 0, -1)[:, None]; post = C[-1] + t1 * np.arange(1, b + 1)[:, None]
    return np.r_[pre, C, post], f


def frame(C):
    from scipy.ndimage import gaussian_filter1d
    T = gaussian_filter1d(np.gradient(C, axis=0), 2.0, axis=0, mode="nearest"); T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-6)
    ez = np.array([1.0, 0, 0]); U = np.zeros_like(C); prev = None
    for i in range(len(C)):
        u = ez - T[i] @ ez * T[i]; nu = np.linalg.norm(u)
        if nu < 0.25 and prev is not None: u = prev - T[i] @ prev * T[i]; nu = np.linalg.norm(u)
        if nu < 1e-6: u = np.array([0, 1.0, 0]) - T[i, 1] * T[i]; nu = np.linalg.norm(u)
        U[i] = u / nu; prev = U[i]
    V = np.cross(T, U); return U, V


def one(cid):
    try: return one_(cid)
    except Exception as e: return cid, f"FAIL {e!r}"[:200], 0, 0


def one_(cid):
    from scipy.ndimage import map_coordinates
    dst = os.path.join(TD, cid + ".npz")
    if os.path.exists(dst): return cid, "skip", 0, 0
    t0 = time.time(); z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb, lo = z["hu"], z["lab"], z["lo"]
    huf = hu.astype(np.float32); sh = np.array(hu.shape); o = np.arange(a.cs) - (a.cs - 1) / 2
    H, M, TAB, OOR, CL = [], [], [], [], []; st = 0
    for k in range(1, 25):
        P = np.argwhere(lb == k).astype(float)
        if len(P) < a.minvox: continue
        r = centerline(P)
        if r is None: continue
        C, nv, nbr = r; n = len(C); Ce, m0 = extend(C); U, V = frame(Ce)
        g = Ce[:, None, None, :] + o[None, :, None, None] * U[:, None, None, :] + o[None, None, :, None] * V[:, None, None, :]
        g = g.transpose(3, 0, 1, 2)
        oor = float(((g < 0) | (g > (sh - 1)[:, None, None, None])).any(0).mean())
        H.append(np.round(map_coordinates(huf, g, order=1, mode="nearest", prefilter=False)).astype(np.int16))
        mk = map_coordinates((lb == k).astype(np.float32), g, order=1, mode="nearest", prefilter=False) >= 0.5
        M.append(np.packbits(mk, axis=-1)); L = len(Ce)
        TAB.append((k, st, L, m0, n, nv)); OOR.append(oor); CL.append((Ce + lo).astype(np.float16)); st += L
    if not TAB: return cid, "norib", 0, time.time() - t0
    tmp = os.path.join(TD, cid + ".tmp.npz")
    np.savez_compressed(tmp, hu=np.concatenate(H), m=np.concatenate(M), tab=np.array(TAB, np.int32), oor=np.array(OOR, np.float32),
                        cl=np.concatenate(CL))
    os.replace(tmp, dst); return cid, "ok", len(TAB), time.time() - t0


if a.qc:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not os.path.exists(os.path.join(TD, a.qc + ".npz")): print(one(a.qc))
    z = np.load(os.path.join(TD, a.qc + ".npz")); tab = z["tab"]; hu = z["hu"]; m = np.unpackbits(z["m"], -1)[..., :a.cs]
    R = len(tab); fig, ax = plt.subplots(R, 2, figsize=(14, 1.3 * R))
    for r, (k, s0, L, m0, n, nv) in enumerate(tab):
        tl = hu[s0:s0 + L]; ml = m[s0:s0 + L]
        ax[r, 0].imshow(tl[:, :, a.cs // 2].T, cmap="gray", vmin=-400, vmax=1200, aspect="auto"); ax[r, 0].set_ylabel(f"k{k}", fontsize=7)
        ax[r, 0].axvline(m0, color="r", lw=.5); ax[r, 0].axvline(m0 + n, color="r", lw=.5)
        ax[r, 1].imshow(tl[:, a.cs // 2, :].T, cmap="gray", vmin=-400, vmax=1200, aspect="auto")
        ax[r, 1].contour(ml[:, a.cs // 2, :].T, [0.5], colors="y", linewidths=.4)
        for q in ax[r]: q.set_xticks([]); q.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(a.out, f"qc_{a.qc}.png"), dpi=70); print("QC", tab.tolist(), z["oor"].round(3).tolist()); sys.exit(0)

if __name__ == "__main__":
    from multiprocessing import Pool
    pi, pn = (int(x) for x in a.part.split("/"))
    ids = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else sorted(f[:-4] for f in os.listdir(a.roi) if f.endswith(".npz") and ".tmp" not in f)
    ids = [i for i in ids if zlib.crc32(i.encode()) % pn == pi and os.path.exists(os.path.join(a.roi, i + ".npz"))]
    if a.limit: ids = ids[:a.limit]
    todo = [i for i in ids if not os.path.exists(os.path.join(TD, i + ".npz"))]
    print(f"{len(todo)}/{len(ids)} to do, jobs {a.jobs}", flush=True); t0 = time.time(); n = 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(a.jobs, maxtasksperchild=100) as p, open(os.path.join(a.out, "status.tsv"), "a") as fh:
        it = p.imap_unordered(one, todo, chunksize=1); nerr = 0
        while True:
            if os.path.exists(os.path.join(a.out, "STOP")): print("STOP seen", flush=True); p.terminate(); break
            try: cid, s, R, dt = it.next(timeout=600)
            except StopIteration: break
            except Exception as e:
                print("ERR", repr(e)[:200], flush=True); nerr += 1
                if nerr >= 20: print("too many errors -> abort", flush=True); p.terminate(); break
                continue
            fh.write(f"{cid}\t{s}\t{R}\t{dt:.2f}\n"); fh.flush(); n += 1
            if n % 500 == 0: print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("TILE_DONE", n, flush=True)
