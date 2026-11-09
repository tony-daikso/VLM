"""Task A: GB mask / crop placement check for E-class FN (report says calcified) -- train/val ids only, no labels/report text read.
For each id: 3 mm prep (hu, units) + TS total 3 mm mask (+ GS6 1.5 mm GB if present) + gs15 1.5 mm crop (if present).
Blob search: HU > 130 connected components (26-conn) within 40 mm of the GB mask, annotated with the TS label under them,
distance to GB (mm), size, max HU; 'plausible' = not bone(+3mm)/vessel/kidney/bowel/stomach/spleen/pancreas and not > 6 mm
inside the liver.  Mask QC: GB HU stats, overlap of the dilated GB with liver/duodenum/colon, GB centroid to liver-surface dist.
PNG: rows = 3 mm W400/L40, 3 mm W200/L50 (axial/coronal through GB centroid; tight 80 mm + wide 120 mm), slices through the
top plausible blob (or dependent GB slice), gs15 1.5 mm crop (axial/coronal W400 & W200).
usage: gbf_render.py ids.txt outdir_png out_csv prepdir1[,prepdir2] maskmap.tsv gs6dir gs15dir"""
import sys, os, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from totalsegmentator.map_to_binary import class_map
CM = class_map["total"]; INV = {v: k for k, v in CM.items()}
def ids_of(*subs): return [k for k, v in CM.items() if any(s in v for s in subs)]
BONE = ids_of("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula", "clavicula", "skull")
VESS = ids_of("aorta", "vena", "portal_vein", "iliac", "heart", "atrial", "pulmonary", "brachioc", "subclavian", "carotid")
OTHER = ids_of("kidney", "spleen", "pancreas", "stomach", "small_bowel", "colon", "esophagus", "adrenal", "urinary_bladder")
LIV, DUO, COL, GBL = INV["liver"], INV["duodenum"], INV["colon"], INV["gallbladder"]
ST26 = np.ones((3, 3, 3), bool)

IDS, PNG, OUT, PREPS, MM, GS6, G15 = sys.argv[1:8]
PREPS = PREPS.split(","); MMAP = dict(l.rstrip("\n").split("\t") for l in open(MM))
os.makedirs(PNG, exist_ok=True)


def load(sid):
    for p in PREPS:
        f = f"{p}/{sid}.npz"
        if os.path.exists(f): z = np.load(f); break
    else: return None
    m = np.asanyarray(nib.load(MMAP[sid]).dataobj).astype(np.int16).transpose(2, 1, 0)
    names = list(z["unit_names"]); gb = z["units"] == names.index("gallbladder") + 1
    src = "ts3"
    f6 = f"{GS6}/{sid}.nii.gz"
    if gb.sum() < 100 and os.path.exists(f6):
        from nibabel.processing import resample_from_to
        seg = nib.load(f6); g = (np.asanyarray(seg.dataobj) == 1).astype(np.float32)
        r = resample_from_to(nib.Nifti1Image(g, seg.affine), (z["units"].shape[::-1], z["affine"]), order=1)
        g6 = (np.asanyarray(r.dataobj) >= .5).transpose(2, 1, 0)
        if g6.sum() >= gb.sum(): gb, src = g6, "gs6"
    return z["hu"].astype(np.float32), m, gb, z["affine"], src


def blobs(hu, m, gb, sp=3.0, maxd=40.0, minvox=2, minmax=160, deep=3.0, maxvol=3000.0):
    """>130 HU components within maxd of GB after removing bone(+1 vox)/vessel/other-organ/deep-liver (> deep mm) voxels;
    plausible stone = max >= minmax and (n >= minvox or max >= 250), volume <= maxvol mm3, not metal (>= 1500 HU)."""
    D = ndimage.distance_transform_edt(~gb) * sp
    L = m == LIV; Ld = ndimage.distance_transform_edt(L) * sp
    excl = ndimage.binary_dilation(np.isin(m, BONE), iterations=1) | np.isin(m, VESS + OTHER + [DUO]) | (Ld > deep)
    excl &= ~gb
    cand = (hu > 130) & (D <= maxd) & ~excl
    lab, n = ndimage.label(cand, ST26); rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        r = lab[sl] == i; idx = np.argwhere(r) + [s.start for s in sl]
        t = tuple(idx.T); labs = m[t]; lv, lc = np.unique(labs, return_counts=True); tl = int(lv[lc.argmax()])
        organ = CM.get(tl, "none") if tl else "none"
        if gb[t].any(): organ = "in_gb_mask"
        mx = float(hu[t].max()); vol = len(idx) * sp ** 3; metal = mx >= 1500
        big = mx >= minmax and (len(idx) >= minvox or mx >= 250)
        k = hu[t].argmax()
        rows.append(dict(n_vox=len(idx), vol_mm3=vol, max_hu=mx, mean_hu=float(hu[t].mean()), dist_gb_mm=float(D[t].min()),
                         organ=organ, in_gb=bool(gb[t].any()), metal=bool(metal),
                         plausible=bool(big and vol <= maxvol and not metal),
                         z=int(idx[k, 0]), y=int(idx[k, 1]), x=int(idx[k, 2]), depth_in_liver_mm=float(Ld[t].max())))
    return pd.DataFrame(rows), D


def maskqc(hu, m, gb, sp=3.0):
    q = dict(gb_vox=int(gb.sum()))
    if gb.sum() < 5: return q
    v = hu[gb]; q.update(gb_med=float(np.median(v)), gb_p10=float(np.percentile(v, 10)), gb_p90=float(np.percentile(v, 90)),
                         gb_max=float(v.max()), gb_n130=int((v > 130).sum()))
    ring = ndimage.binary_dilation(gb, iterations=1) & ~gb; rl = m[ring]
    q["ring_frac_liver"] = float((rl == LIV).mean()); q["ring_frac_duod"] = float((rl == DUO).mean()); q["ring_frac_colon"] = float((rl == COL).mean())
    q["ring_frac_fat"] = float(((hu[ring] < -30)).mean())
    L = m == LIV
    if L.sum() > 1000:
        c = np.round(ndimage.center_of_mass(gb)).astype(int); Lout = ndimage.distance_transform_edt(~L) * sp
        q["cen_dist_to_liver_mm"] = float(Lout[tuple(c)])
        lz = np.where(L.any((1, 2)))[0]; q["cen_z_frac_liver"] = float((c[0] - lz[0]) / max(lz[-1] - lz[0], 1))
        q["gb_in_liver_frac"] = float(L[gb].mean())
    return q


def win(a, W, Lv): return np.clip((a - (Lv - W / 2)) / W, 0, 1)


def panel(ax, img, msk, W, Lv, title, ext, mark=None, ring=None):
    # img rows = superior up (axial: anterior up). display radiological: patient right on viewer left -> x increasing = R? RAS +x = L
    ax.imshow(win(img, W, Lv), cmap="gray", origin="lower", extent=ext, interpolation="nearest")
    if msk is not None and msk.any(): ax.contour(msk.astype(float), levels=[.5], colors="lime", linewidths=.8, origin="lower", extent=ext)
    if ring is not None and ring.any(): ax.contour(ring.astype(float), levels=[.5], colors="cyan", linewidths=.5, linestyles="dotted", origin="lower", extent=ext)
    if mark is not None:
        for (u, v, c) in mark: ax.plot(u, v, "o", mfc="none", mec=c, ms=9, mew=1.2)
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def sl2d(a, axis, i):
    i = int(np.clip(i, 0, a.shape[axis] - 1))
    return np.take(a, i, axis=axis)


def render(sid, hu, m, gb, A, B, D, g15, info):
    sp = 3.0; c = np.array(ndimage.center_of_mass(gb)) if gb.any() else np.array(hu.shape) / 2
    fig, axs = plt.subplots(4, 4, figsize=(13, 13.4)); fig.patch.set_facecolor("white")
    ring40 = D <= 40
    pl = B[B.plausible] if len(B) else B
    def views(row, W, Lv, cz, cy, cx, tag, marks):
        # axial: array [y, x]; display x flipped so patient right (low RAS x? RAS +x = patient Left) appears on viewer's left
        for j, (half, nm) in enumerate(((40, "80mm"), (60, "120mm"))):
            ax_img = sl2d(hu, 0, cz); ax_m = sl2d(gb, 0, cz); ax_r = sl2d(ring40, 0, cz)
            ext = [0, hu.shape[2] * sp, 0, hu.shape[1] * sp]
            mk = [(x * sp + 1.5, y * sp + 1.5, col) for (z, y, x, col) in marks if abs(z - cz) <= 1]
            panel(axs[row, j], ax_img, ax_m, W, Lv, f"{tag} axial W{W}/L{Lv} {nm}", ext, mk, ax_r if j else None)
            axs[row, j].set_xlim(cx * sp + half, cx * sp - half); axs[row, j].set_ylim(cy * sp - half, cy * sp + half)
            co_img = sl2d(hu, 1, cy); co_m = sl2d(gb, 1, cy); co_r = sl2d(ring40, 1, cy)
            ext = [0, hu.shape[2] * sp, 0, hu.shape[0] * sp]
            mk = [(x * sp + 1.5, z * sp + 1.5, col) for (z, y, x, col) in marks if abs(y - cy) <= 1]
            panel(axs[row, j + 2], co_img, co_m, W, Lv, f"{tag} coronal W{W}/L{Lv} {nm}", ext, mk, co_r if j else None)
            axs[row, j + 2].set_xlim(cx * sp + half, cx * sp - half); axs[row, j + 2].set_ylim(cz * sp - half, cz * sp + half)
    marks = [(int(r.z), int(r.y), int(r.x), "red" if r.plausible else "deepskyblue") for r in B.itertuples() if r.plausible or r.metal] if len(B) else []
    cz, cy, cx = np.round(c).astype(int)
    views(0, 400, 40, cz, cy, cx, "3mm GB-cen", marks)
    views(1, 200, 50, cz, cy, cx, "3mm GB-cen", marks)
    if len(pl):
        t = pl.sort_values("max_hu", ascending=False).iloc[0]; bz, by, bx = int(t.z), int(t.y), int(t.x)
        tag = f"top blob {t.max_hu:.0f}HU d={t.dist_gb_mm:.0f}mm {t.organ}"
    else:
        g = np.argwhere(gb); bz, by, bx = (np.round(g[g[:, 1] <= np.percentile(g[:, 1], 20)].mean(0)).astype(int) if len(g) else (cz, cy, cx))
        tag = "dependent GB (no blob)"
    # row 2: through blob, W400 tight + W200 tight axial / coronal
    for j, (W, Lv) in enumerate(((400, 40), (200, 50))):
        ext = [0, hu.shape[2] * sp, 0, hu.shape[1] * sp]
        mk = [(x * sp + 1.5, y * sp + 1.5, col) for (z, y, x, col) in marks if abs(z - bz) <= 1]
        panel(axs[2, j], sl2d(hu, 0, bz), sl2d(gb, 0, bz), W, Lv, f"{tag[:38]}\naxial W{W}/L{Lv}", ext, mk)
        axs[2, j].set_xlim(bx * sp + 45, bx * sp - 45); axs[2, j].set_ylim(by * sp - 45, by * sp + 45)
        ext = [0, hu.shape[2] * sp, 0, hu.shape[0] * sp]
        mk = [(x * sp + 1.5, z * sp + 1.5, col) for (z, y, x, col) in marks if abs(y - by) <= 1]
        panel(axs[2, j + 2], sl2d(hu, 1, by), sl2d(gb, 1, by), W, Lv, f"coronal W{W}/L{Lv}", ext, mk)
        axs[2, j + 2].set_xlim(bx * sp + 45, bx * sp - 45); axs[2, j + 2].set_ylim(bz * sp - 45, bz * sp + 45)
    # row 3: gs15 crop
    if g15 is not None:
        h, gm, B15 = g15; S = h.shape[0]; gc = np.round(ndimage.center_of_mass(gm)).astype(int) if gm.any() else np.array([S // 2] * 3)
        p15 = B15[B15.plausible] if len(B15) else B15
        m15 = [(int(r.z), int(r.y), int(r.x), "red") for r in p15.itertuples()]
        if len(p15):
            t = p15.sort_values("max_hu").iloc[-1]; tz, ty = int(t.z), int(t.y); tt = f"gs15 blob {t.max_hu:.0f}HU d={t.dist_gb_mm:.0f}mm"
        else:
            tz, ty, tt = gc[0], gc[1], "gs15 GB-cen"
        ext = [0, S * 1.5, 0, S * 1.5]
        for j, (zz, yy, nm) in enumerate(((gc[0], gc[1], "gs15 GB-cen"), (tz, ty, tt))):
            mk = [(x * 1.5 + .75, y * 1.5 + .75, col) for (z, y, x, col) in m15 if abs(z - zz) <= 2]
            panel(axs[3, 2 * j], sl2d(h, 0, zz), sl2d(gm, 0, zz), 400, 40, f"{nm} 1.5mm axial W400/L40", ext, mk)
            axs[3, 2 * j].set_xlim(S * 1.5, 0)
            mk = [(x * 1.5 + .75, z * 1.5 + .75, col) for (z, y, x, col) in m15 if abs(y - yy) <= 2]
            panel(axs[3, 2 * j + 1], sl2d(h, 1, yy), sl2d(gm, 1, yy), 200, 50, f"{nm} 1.5mm coronal W200/L50", ext, mk)
            axs[3, 2 * j + 1].set_xlim(S * 1.5, 0)
    else:
        for j in range(4): axs[3, j].axis("off")
        axs[3, 0].text(.05, .5, "no gs15 crop", transform=axs[3, 0].transAxes)
    fig.suptitle(f"{sid} | {info}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(f"{PNG}/{sid}.png", dpi=85); plt.close(fig)


def g15load(sid):
    f = f"{G15}/{sid}.npz"
    if not os.path.exists(f): return None, {}, None
    c = np.load(f, allow_pickle=True)
    if not bool(c["present"][0]): return None, {}, None
    h = c["hu"][0].astype(np.float32); gm = np.unpackbits(c["gbmask"])[:h.size].reshape(h.shape).astype(bool)
    q = {"g15_method": str(c["method"][0]), "g15_gb_vox": int(gm.sum())}
    S, SP = h.shape[0], 1.5; cw = c["center_world"]; off = (np.arange(S) - (S - 1) / 2) * SP
    X, Y, Z = np.meshgrid(cw[0] + off, cw[1] + off, cw[2] + off, indexing="ij")
    G = np.stack([X, Y, Z, np.ones_like(X)], 0).reshape(4, -1)
    mi = nib.load(MMAP[sid]); v = np.linalg.inv(mi.affine) @ G
    lab = ndimage.map_coordinates(np.asanyarray(mi.dataobj).astype(np.int16), v[:3], order=0, mode="constant", cval=0).reshape(S, S, S).transpose(2, 1, 0)
    q["g15_gb_ts_overlap"] = float((lab[gm] == GBL).mean()) if gm.any() else np.nan
    B = pd.DataFrame()
    if gm.any():
        B, D = blobs(h, lab, gm, sp=1.5, maxd=40.0, minvox=2, minmax=160, deep=4.5)
        pl = B[B.plausible] if len(B) else B
        q.update(g15_n_plaus_in_gb=int(pl.in_gb.sum()) if len(pl) else 0,
                 g15_n_plaus_0_15=int(((pl.dist_gb_mm > 0) & (pl.dist_gb_mm <= 15)).sum()) if len(pl) else 0,
                 g15_n_plaus_15_40=int((pl.dist_gb_mm > 15).sum()) if len(pl) else 0,
                 g15_top_hu=float(pl.max_hu.max()) if len(pl) else np.nan,
                 g15_top_dist=float(pl.sort_values("max_hu").iloc[-1].dist_gb_mm) if len(pl) else np.nan,
                 g15_top_organ=str(pl.sort_values("max_hu").iloc[-1].organ) if len(pl) else "",
                 g15_n_metal=int(B.metal.sum()) if len(B) else 0)
    return (h, gm, B), q, lab


rows = []; allb = []
for sid in [l.strip() for l in open(IDS) if l.strip()]:
    L = load(sid)
    if L is None: rows.append(dict(study_id=sid, err="no_prep")); continue
    hu, m, gb, A, src = L
    B, D = blobs(hu, m, gb); q = maskqc(hu, m, gb); g15, q15, _ = g15load(sid)
    pl = B[B.plausible] if len(B) else B
    r = dict(study_id=sid, gb_src=src, **q, **q15, n_blob=len(B), n_metal=int(B.metal.sum()) if len(B) else 0, n_plaus=len(pl),
             n_plaus_in_gb=int(pl.in_gb.sum()) if len(pl) else 0,
             n_plaus_0_15=int(((pl.dist_gb_mm > 0) & (pl.dist_gb_mm <= 15)).sum()) if len(pl) else 0,
             n_plaus_15_40=int((pl.dist_gb_mm > 15).sum()) if len(pl) else 0,
             top_plaus_hu=float(pl.max_hu.max()) if len(pl) else np.nan,
             top_plaus_dist=float(pl.sort_values("max_hu").iloc[-1].dist_gb_mm) if len(pl) else np.nan,
             top_plaus_organ=str(pl.sort_values("max_hu").iloc[-1].organ) if len(pl) else "")
    rows.append(r)
    if g15 is not None and len(g15[2]): allb.append(g15[2].assign(study_id=sid, res='1.5mm'))
    if len(B): allb.append(B.assign(study_id=sid, res='3mm'))
    info = f"GB {src} {q.get('gb_vox')} vox med {q.get('gb_med', np.nan):.0f} HU | plausible >130HU blobs: in-GB {r['n_plaus_in_gb']}, 0-15mm {r['n_plaus_0_15']}, 15-40mm {r['n_plaus_15_40']} | red=stone-like blob, blue=metal; green=GB mask, cyan=40mm"
    render(sid, hu, m, gb, A, B, D, g15, info)
    print(sid, json.dumps(r, default=str), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
if allb: pd.concat(allb).to_csv(OUT.replace(".csv", "_blobs.csv"), index=False)
