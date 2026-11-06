"""QC renders for the RLQ crop centre (no IDs on images). For each case: axial slice through the centre (full FOV) with
colon (green), small bowel (yellow), iliopsoas_right (magenta) contours, centre (red +) and the 192 mm crop box (cyan);
plus the coronal-like slice (fixed AP index) through the centre; plus the crop-level pericecal region r1 (orange).
Usage: rlq_qc.py ids.txt crop_dir out_dir prep1:mask1 [...]   -> out_dir/qc_XX.png (+ qc_index.csv mapping XX->id,
kept off the images)."""
import sys, os, numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ids, cdir, odir = sys.argv[1:4]
pairs = [tuple(p.split(":")) for p in sys.argv[4:]]
os.makedirs(odir, exist_ok=True)
L = [l.strip() for l in open(ids) if l.strip()]
idx = open(f"{odir}/qc_index.csv", "w"); idx.write("png,study_id,method\n")


def sl(a, ax, i):
    return np.take(a, int(np.clip(i, 0, a.shape[ax] - 1)), axis=ax)


for n, sid in enumerate(L):
    z = np.load(f"{cdir}/{sid}.npz")
    if not bool(z["present"]):
        continue
    src = next(((p, mk) for p, mk in pairs if os.path.exists(f"{p}/{sid}.npz") and os.path.exists(f"{mk}/{sid}.nii.gz")), None)
    hu = np.load(f"{src[0]}/{sid}.npz")["hu"]; m = np.asarray(nib.load(f"{src[1]}/{sid}.nii.gz").dataobj).astype(np.uint8).T
    c = z["center"]; st = z["start"]; lr, sr, ap, sa, ss = [int(v) for v in z["orient"]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    # axial: fix axis 0; display rows=ap (anterior up), cols=lr (patient right on image left, radiological)
    def show(ax, a2, m2, title, rows_ax, cols_ax, flip_r, flip_c, cpt, box):
        img = np.clip((a2 + 160) / 400, 0, 1)
        if flip_r: img, m2 = img[::-1], m2[::-1]
        if flip_c: img, m2 = img[:, ::-1], m2[:, ::-1]
        ax.imshow(img, cmap="gray")
        for lab, col in ((20, "lime"), (18, "yellow"), (89, "magenta")):
            if (m2 == lab).any(): ax.contour(m2 == lab, levels=[0.5], colors=col, linewidths=0.8)
        H, W = img.shape
        r, cc = cpt
        r = H - 1 - r if flip_r else r; cc = W - 1 - cc if flip_c else cc
        ax.plot(cc, r, "r+", ms=16, mew=2)
        (r0, c0) = box
        r0 = H - 64 - r0 if flip_r else r0; c0 = W - 64 - c0 if flip_c else c0
        ax.add_patch(plt.Rectangle((c0, r0), 64, 64, fill=False, ec="cyan", lw=1))
        ax.set_title(title); ax.axis("off")
    # axial (rows = AP, cols = LR)
    a2 = sl(hu, 0, c[0]); m2 = sl(m, 0, c[0])
    A2, M2 = (a2, m2) if ap == 1 else (a2.T, m2.T)
    show(axs[0], A2, M2, f"axial ({z['method']})", ap, lr, sa > 0, sr > 0, (c[ap], c[lr]), (st[ap], st[lr]))
    # coronal: fix AP axis; rows = SI (axis 0), cols = LR
    a2 = sl(hu, ap, c[ap]); m2 = sl(m, ap, c[ap])
    show(axs[1], a2, m2, "coronal", 0, lr, ss > 0, sr > 0, (c[0], c[lr]), (st[0], st[lr]))
    # crop-level axial with r1 pericecal region
    h = z["hu"]; R = z["regions"]; k = 32
    a3 = h[k]; r1 = R[1][k]; r2 = R[2][k]
    if ap != 1: a3, r1, r2 = a3.T, r1.T, r2.T
    img = np.clip((a3 + 160) / 400, 0, 1); fr, fc = sa > 0, sr > 0
    if fr: img, r1, r2 = img[::-1], r1[::-1], r2[::-1]
    if fc: img, r1, r2 = img[:, ::-1], r1[:, ::-1], r2[:, ::-1]
    axs[2].imshow(img, cmap="gray")
    if r1.any(): axs[2].contour(r1, levels=[0.5], colors="orange", linewidths=0.8)
    if r2.any(): axs[2].contour(r2, levels=[0.5], colors="lime", linewidths=0.8)
    axs[2].plot(32, 32, "r+", ms=16, mew=2); axs[2].set_title("crop 192 mm: r1 pericecal (orange), r2 cecum+TI (green)"); axs[2].axis("off")
    fig.suptitle(f"QC {n:02d}   patient right = image left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{odir}/qc_{n:02d}.png", dpi=80); plt.close(fig)
    idx.write(f"qc_{n:02d}.png,{sid},{z['method']}\n")
print("QC_DONE")
