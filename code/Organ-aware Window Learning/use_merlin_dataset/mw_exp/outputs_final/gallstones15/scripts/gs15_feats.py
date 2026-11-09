"""Collect the 1.5 mm GB HU features + centre method from crops/*.npz -> ROOT/feats.csv (incremental: keeps rows already
collected). Columns f15_<name>, method15. Usage: gs15_feats.py ROOT"""
import sys, os, glob, numpy as np, pandas as pd
R = sys.argv[1]; fo = f"{R}/feats.csv"
old = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame({"study_id": []})
have = set(old.study_id); rows = []
for f in sorted(glob.glob(f"{R}/crops/AC*.npz")):
    sid = os.path.basename(f)[:-4]
    if sid in have:
        continue
    try:
        z = np.load(f)
        r = {"study_id": sid, "method15": str(z["method"][0]) if "method" in z else ""}
        if "feat" in z:
            r.update({f"f15_{k}": float(v) for k, v in zip(z["feat_names"], z["feat"])})
        rows.append(r)
    except Exception as e:  # noqa: BLE001
        print("ERR", sid, repr(e)[:100], flush=True)
d = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
d.to_csv(fo, index=False); print("feats", len(d), "new", len(rows), flush=True)
