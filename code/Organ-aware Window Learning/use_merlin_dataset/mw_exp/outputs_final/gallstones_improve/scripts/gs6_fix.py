"""GS6 step 4 (CPU): put the full-resolution GB (gs6_seg.py, 1.5 mm) into the 3 mm prep/mask of the case.
GB(1.5 mm) -> 3 mm prep grid by linear resampling of the binary mask, voxel fraction >= 0.5. Old 3 mm GB voxels are cleared,
new GB voxels get the gallbladder label (prep 'units' and TS 'total' mask); everything else (liver, HU) is unchanged.
Writes out_dir/prep/{id}.npz, out_dir/masks/{id}.nii.gz and out_dir/fix_log.csv (old/new 3 mm GB voxels).
Feature scripts (gs_hu.py, ag_feat.py, gs1_feat.py) and gs5_crop.py are then run on these dirs.
usage: gs6_fix.py ids.txt prep_dir[,..] mask_dir[,..] seg_dir out_dir"""
import sys, os, numpy as np, pandas as pd, nibabel as nib
from nibabel.processing import resample_from_to
from totalsegmentator.map_to_binary import class_map
GBT = {v: k for k, v in class_map["total"].items()}["gallbladder"]
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
PD, MD, SD, OD = sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4], sys.argv[5]
os.makedirs(f"{OD}/prep", exist_ok=True); os.makedirs(f"{OD}/masks", exist_ok=True)
def find(s, dirs, ext): return next((f"{d}/{s}{ext}" for d in dirs if os.path.exists(f"{d}/{s}{ext}")), None)
rows = []
for s in ids:
    fs, fp, fm = f"{SD}/{s}.nii.gz", find(s, PD, ".npz"), find(s, MD, ".nii.gz")
    if not (os.path.exists(fs) and fp and fm): rows.append(dict(study_id=s, status="nofile")); continue
    if os.path.exists(f"{OD}/prep/{s}.npz"): continue
    z = dict(np.load(fp)); names = list(z["unit_names"]); gi = names.index("gallbladder") + 1
    seg = nib.load(fs); g15 = (np.asanyarray(seg.dataobj) == 1)
    shp_xyz = z["units"].shape[::-1]
    r = resample_from_to(nib.Nifti1Image(g15.astype(np.float32), seg.affine), (shp_xyz, z["affine"]), order=1, cval=0.0)
    new = (np.asanyarray(r.dataobj) >= 0.5).transpose(2, 1, 0)          # -> (z,y,x)
    old = z["units"] == gi
    u = z["units"].copy(); u[old] = 0; u[new] = gi; z["units"] = u
    np.savez_compressed(f"{OD}/prep/{s}.npz", **z)
    mi = nib.load(fm); m = np.asanyarray(mi.dataobj).astype(np.uint8).copy()   # (x,y,z)
    newx = new.transpose(2, 1, 0); m[m == GBT] = 0; m[newx] = GBT
    nib.save(nib.Nifti1Image(m, mi.affine), f"{OD}/masks/{s}.nii.gz")
    rows.append(dict(study_id=s, status="ok", gb3_old=int(old.sum()), gb3_new=int(new.sum()), gb15=int(g15.sum())))
f = f"{OD}/fix_log.csv"
pd.DataFrame(rows).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
print("FIX_DONE", pd.DataFrame(rows).status.value_counts().to_dict() if rows else {}, flush=True)
