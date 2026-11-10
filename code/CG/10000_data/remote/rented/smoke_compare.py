"""ON CT-rate: compare L40S smoke outputs (~/OWL/cg10k_infer/smoke_l40s/{v15,nc3}) with CT-rate references.
usage: python smoke_compare.py V15_ID1,V15_ID2 MM3_ID   (3 mm reference = owl/main)"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

R = Path("/media/daikso/CTDATA/cg10k")
S = Path.home() / "OWL/cg10k_infer/smoke_l40s"


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def per_row_cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    keep = (np.abs(a).sum(1) > 0) | (np.abs(b).sum(1) > 0)
    a, b = a[keep], b[keep]
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(c.min()) if len(c) else float("nan"), float(c.mean()) if len(c) else float("nan")


v15, mm3 = sys.argv[1].split(","), sys.argv[2]
for c in v15:
    a, b = np.load(R / "owl_v15/main/emb15" / f"{c}.npz"), np.load(S / "v15/emb15" / f"{c}.npz")
    print(f"v15 {c}: present equal={np.array_equal(a['present'], b['present'])} "
          f"shape15 {a['shape15'].tolist()} vs {b['shape15'].tolist()}")
    for k in ("emb", "emb_max"):
        mn, me = per_row_cos(a[k], b[k])
        print(f"   {k}: global cos {cos(a[k], b[k]):.6f}  per-unit/window/stage cos min {mn:.6f} mean {me:.6f}  "
              f"max|diff| {np.nanmax(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32))):.4g}")
    sa, sb = np.load(R / "owl_v15/main/seg15" / f"{c}.npz"), np.load(S / "v15/seg15" / f"{c}.npz")
    same = sa["lab"].shape == sb["lab"].shape and np.array_equal(sa["off"], sb["off"])
    agree = float((sa["lab"] == sb["lab"]).mean()) if same else float("nan")
    fg = (sa["lab"] > 0) | (sb["lab"] > 0) if same else None
    dice_fg = float(2 * ((sa["lab"] > 0) & (sb["lab"] > 0)).sum() / ((sa["lab"] > 0).sum() + (sb["lab"] > 0).sum())) if same else float("nan")
    print(f"   seg15: same crop {same}, voxel agreement {agree:.6f}, fg dice {dice_fg:.6f}, "
          f"exact {same and np.array_equal(sa['lab'], sb['lab'])}, labels on fg agree "
          f"{float((sa['lab'][fg] == sb['lab'][fg]).mean()) if same else float('nan'):.6f}")

ma = np.asanyarray(nib.load(str(R / "owl/main/masks" / f"{mm3}.nii.gz")).dataobj)
mb = np.asanyarray(nib.load(str(S / "nc3/masks" / f"{mm3}.nii.gz")).dataobj)
fg = (ma > 0) | (mb > 0)
print(f"3mm {mm3}: mask shape {ma.shape} vs {mb.shape}, exact {np.array_equal(ma, mb)}, "
      f"voxel agreement {float((ma == mb).mean()):.6f}, fg label agreement {float((ma[fg] == mb[fg]).mean()):.6f}")
for sub in ("emb", "emb_bones"):
    for d in sorted((R / "owl/main" / sub).iterdir()):
        f2 = S / "nc3" / sub / d.name / f"{mm3}.npz"
        if not f2.exists():
            continue
        a, b = np.load(d / f"{mm3}.npz"), np.load(f2)
        ks = [k for k in a.files if a[k].dtype.kind == "f" and a[k].ndim >= 2]
        msg = []
        for k in ks:
            mn, me = per_row_cos(a[k], b[k])
            msg.append(f"{k}: cos {cos(a[k], b[k]):.6f} min {mn:.6f}")
        print(f"   {sub}/{d.name}: " + "; ".join(msg))
