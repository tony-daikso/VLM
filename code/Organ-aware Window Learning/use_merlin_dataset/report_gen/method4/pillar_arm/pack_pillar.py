#!/usr/bin/env python3
"""Option B (arm P_gt), step 0 (CT-rate, numpy only, CPU; run with nice 19 / ionice -c3): pack the finished Pillar-0 npz
files into ONE memmap-able store, so A6000 gets a single grid16 array instead of 20k npz files.

<out>/ids.txt        N case ids (row order; train+val of split.csv in file order, or --ids-file); fixed at creation
<out>/grid16.npy     (N, 4096, 384) float16   grid16 tokens, raster (z, y, x) of the 384^3 input (open_memmap)
<out>/units.npy      (N, 52, 768)   float16   pillar_units.UNITS composites: vec16 | vec64 (exact weighted merge)
<out>/uinfo.npy      (N, 52, 3)     float32   present, crop coverage, log1p(mL)/5
<out>/zvalid.npy     (N, 2)         int16     first / last grid16 z-cell holding image data (rest = z padding)
<out>/done.npy       (N,)           uint8     1 = row filled from a finished <id>.npz
<out>/pack_meta.json counts, unit list, source, bad files
Incremental: each run fills only rows not done yet whose <id>.npz exists (the atomic rename = finished marker;
*.tmp.npz ignored), so it can run while the extractor is still going. Test ids are refused (the extractor never writes
them either). --global also stores the 1152-d global vector (not used by the P_gt adapter; off by default).
usage: OMP_NUM_THREADS=1 nice -n 19 ionice -c3 python pack_pillar.py --feats /media/daikso/Tony/pillar/merlin/feats \
          --out /media/daikso/Tony/pillar/merlin/p_arm_store [--ids-file ids.txt] [--max 0]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pillar_units import INFO_DIM, N_UNITS, UNIT_DIM, UNIT_NAMES, UNITS, check_classmap, merge_units, z_valid_cells  # noqa: E402

SPLIT = "/media/daikso/Tony/OWL/data_h/split.csv"


def open_store(out, ids, with_global):
    from numpy.lib.format import open_memmap
    n = len(ids)
    spec = {"grid16": ((n, 4096, 384), np.float16), "units": ((n, N_UNITS, UNIT_DIM), np.float16),
            "uinfo": ((n, N_UNITS, INFO_DIM), np.float32), "zvalid": ((n, 2), np.int16), "done": ((n,), np.uint8)}
    if with_global:
        spec["global"] = ((n, 1152), np.float16)
    arr = {}
    for k, (shape, dt) in spec.items():
        p = out / f"{k}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r+")
            assert a.shape == shape and a.dtype == dt, (k, a.shape, shape)
        else:
            a = open_memmap(p, mode="w+", dtype=dt, shape=shape)   # sparse until written
        arr[k] = a
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--ids-file", default="", help="row order for a new store (default: train+val of --split)")
    ap.add_argument("--classmap", default=str(Path(__file__).resolve().parent / "ts_total_classmap.json"))
    ap.add_argument("--global", dest="with_global", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="fill at most this many new rows (0 = all)")
    a = ap.parse_args()
    if Path(a.classmap).exists():
        check_classmap(a.classmap)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = {r["study_id"]: r["split"] for r in csv.DictReader(open(a.split))}
    if (out / "ids.txt").exists():
        ids = (out / "ids.txt").read_text().split()
    else:
        ids = (Path(a.ids_file).read_text().split() if a.ids_file
               else [s for s, sp in split.items() if sp in ("train", "val")])
        bad = [i for i in ids if split.get(i) not in ("train", "val")]
        if bad:
            sys.exit(f"{len(bad)} ids are not train/val (e.g. {bad[0]}) -- test is never packed")
        (out / "ids.txt").write_text("\n".join(ids) + "\n")
    arr = open_store(out, ids, a.with_global)
    done = arr["done"]
    todo = [i for i in range(len(ids)) if not done[i] and (Path(a.feats) / f"{ids[i]}.npz").exists()]
    if a.max:
        todo = todo[:a.max]
    print(f"store {out}: {len(ids)} rows, {int(done.sum())} done, {len(todo)} new finished npz", flush=True)
    meta_p = out / "pack_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"bad": {}, "mask_missing": []}
    t0 = time.time()
    for j, i in enumerate(todo):
        cid = ids[i]
        try:
            z = np.load(Path(a.feats) / f"{cid}.npz")
            g = z["grid16"]
            if g.shape != (4096, 384) or not np.isfinite(g).all():
                meta["bad"][cid] = f"grid16 {g.shape} finite={bool(np.isfinite(g).all())}"
                continue
            u, info = merge_units(z["vec16"], z["vec64"], z["w16"], z["w64"], z["vox_crop"], z["vox_total"])
            if not (np.isfinite(u).all() and np.isfinite(info).all()):
                meta["bad"][cid] = "units not finite"
                continue
            m = json.loads(str(z["meta"]))
            if not m.get("mask_ok", True):
                meta["mask_missing"].append(cid)
            arr["grid16"][i] = g
            arr["units"][i] = u.astype(np.float16)
            arr["uinfo"][i] = info
            arr["zvalid"][i] = z_valid_cells(m["plan"])
            if a.with_global:
                arr["global"][i] = z["global"]
            done[i] = 1
        except Exception as e:  # noqa: BLE001  (truncated / being replaced -> next run)
            print("skip", cid, repr(e), flush=True)
        if j % 500 == 0 or j == len(todo) - 1:
            for v in arr.values():
                v.flush()
            print(f"  {j + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    for v in arr.values():
        v.flush()
    meta.update(n=len(ids), n_done=int(done.sum()), units=[list(x[:2]) + [x[2]] for x in UNITS],
                unit_names=UNIT_NAMES, feats=str(a.feats), time=time.strftime("%Y-%m-%d %H:%M"),
                n_split={s: sum(split.get(i) == s for i in ids) for s in ("train", "val")})
    meta["mask_missing"] = sorted(set(meta["mask_missing"]))
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"done: {meta['n_done']}/{len(ids)} rows filled, {len(meta['bad'])} bad, "
          f"{len(meta['mask_missing'])} without TS mask", flush=True)


if __name__ == "__main__":
    main()
