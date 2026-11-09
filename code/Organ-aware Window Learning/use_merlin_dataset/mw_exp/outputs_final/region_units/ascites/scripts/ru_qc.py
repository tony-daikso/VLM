"""QC montage for ru_crop.py crops: per case one PNG (no study IDs on the image; mapping in qc_index.csv).
Rows = crops (K); columns = axial / coronal / sagittal-in-AP slice through the crop centre; region contours r0..r3.
Usage: ru_qc.py crops_dir ids.txt out_dir [n=20]"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
cd, ids, od = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
os.makedirs(od, exist_ok=True)
L = [l.strip() for l in open(ids) if l.strip()][:n]
COL = ["red", "lime", "cyan", "yellow"]
idx = open(f"{od}/qc_index.csv", "w"); idx.write("qc,study_id,methods,present\n")
for q, sid in enumerate(L):
    z = np.load(f"{cd}/{sid}.npz")
    if "hu" not in z:
        idx.write(f"{q},{sid},noorient,\n"); continue
    H = z["hu"]; R = np.unpackbits(z["regions"])[:np.prod(z["rshape"])].reshape(z["rshape"]).astype(bool)
    K, S = H.shape[0], H.shape[1]; o = z["orient"]; ap, lr = int(o[2]), int(o[0]); c = S // 2
    fig, axs = plt.subplots(K, 3, figsize=(9, 3 * K + 0.4))
    axs = np.atleast_2d(axs)
    for k in range(K):
        for j, (nm, sl) in enumerate((("axial", lambda a: a[c]), ("coronal", lambda a: a.take(c, axis=ap)), ("sagittal", lambda a: a.take(c, axis=lr)))):
            ax = axs[k, j]
            img = sl(H[k]).astype(float)
            if j > 0: img = img[::-1] if int(o[4]) > 0 else img
            ax.imshow(np.clip((img + 160) / 400, 0, 1), cmap="gray")
            for r in range(R.shape[1]):
                mm = sl(R[k, r]).astype(float)
                if j > 0 and int(o[4]) > 0: mm = mm[::-1]
                if mm.any(): ax.contour(mm, levels=[0.5], colors=COL[r], linewidths=0.7)
            ax.plot(c, c if j == 0 else (S - 1 - c if int(o[4]) > 0 else c), "m+", ms=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"crop{k} {str(z['method'][k])} {nm}", fontsize=7)
    fig.suptitle(f"QC {q:02d}  r0 red, r1 green, r2 cyan, r3 yellow", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{od}/qc_{q:02d}.png", dpi=80); plt.close(fig)
    idx.write(f"{q},{sid},{'|'.join(map(str, z['method']))},{'|'.join(map(str, z['present'].astype(int)))}\n")
print("QC_DONE", len(L))
