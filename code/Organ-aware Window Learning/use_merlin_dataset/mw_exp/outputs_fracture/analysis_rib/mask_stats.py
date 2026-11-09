"""Per-case FOV / rib / pelvis mask statistics from TS total 3 mm masks (train+val ids only; test ids skipped)."""
import sys, numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from joblib import Parallel, delayed
H = Path.home() / "Desktop/Tony/VLM/OWL"
MASK = H / "mw_exp/outputs/masks"
OUT = H / "mw_exp/outputs_fracture/analysis_rib"
spl = pd.read_csv(H / "data/split.csv").set_index("study_id")["split"]
keep = sorted(set(spl.index[spl.isin(["train", "val"])]) & {p.name[:-7] for p in MASK.glob("*.nii.gz")})
RIB = {f"rib_{s}_{i}": (92 if s == "left" else 104) + i - 1 for s in ("left", "right") for i in range(1, 13)}
OTH = {"lung_ul_l": 10, "lung_ll_l": 11, "lung_ul_r": 12, "lung_ml_r": 13, "lung_ll_r": 14, "heart": 51, "sacrum": 25,
       "femur_l": 75, "femur_r": 76, "hip_l": 77, "hip_r": 78, "T1": 43, "T4": 40, "T8": 36, "T12": 32, "L5": 27,
       "sternum": 116, "scapula_l": 71, "scapula_r": 72, "bladder": 21}
def one(cid):
    try:
        m = nib.load(MASK / f"{cid}.nii.gz")
        a = np.asarray(m.dataobj)
        ax = nib.aff2axcodes(m.affine)
        zax = [i for i, c in enumerate(ax) if c in "SI"][0]
        a = np.moveaxis(a, zax, 2)
        if ax[zax] == "I":
            a = a[:, :, ::-1]                     # index increases superiorly
        nz = a.shape[2]
        r = {"id": cid, "nz": nz, "z_mm": nz * float(m.header.get_zooms()[zax])}
        # per-slice label presence -> z-range of each label
        cnt = np.bincount(a.ravel(), minlength=128)
        zs = {}
        for name, lab in {**RIB, **OTH}.items():
            r[f"v_{name}"] = int(cnt[lab])
            if cnt[lab]:
                zz = np.where((a == lab).any((0, 1)))[0]
                r[f"zlo_{name}"], r[f"zhi_{name}"] = int(zz[0]), int(zz[-1])
                r[f"top_{name}"] = int(zz[-1] >= nz - 1)       # touches the superior edge
                r[f"bot_{name}"] = int(zz[0] <= 0)             # touches the inferior edge
        return r
    except Exception as e:
        return {"id": cid, "err": str(e)[:100]}
print(len(keep), "train/val masks", flush=True)
res = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6, batch_size=32, verbose=5)(delayed(one)(c) for c in keep)
df = pd.DataFrame(res).set_index("id")
df["split"] = spl.loc[df.index]
assert set(df.split) <= {"train", "val"}
df.to_csv(OUT / "mask_fov_stats.csv")
print(df.shape, df.get("err", pd.Series(dtype=str)).notna().sum())
