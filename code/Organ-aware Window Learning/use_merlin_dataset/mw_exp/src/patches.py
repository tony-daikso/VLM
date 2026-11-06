"""Step 1b: preprocess (same as TotalSegmentator) + organ units + organ-centred patch coordinates.

Image: nib.load -> as_closest_canonical -> change_spacing(model spacing, order=1, int32)   (= TS)
       -> array transposed (2,1,0) to nnU-Net order (= NibabelIOWithReorient).
Mask : canonical, then nearest-neighbour onto the image grid if grids differ.
Outputs:
  outputs/prep/{id}.npz      hu (int16, nnU-Net order), units (uint8, 0=bg, k=unit k-1), affine
  outputs/patches/{id}.json  per-unit presence, voxels, centroid, bbox, tiles
  outputs/mask_stats.csv     (written by --stats)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from common import case_ids, load_cfg, log_progress, read_plans, shard, unit_names


def label_lookup(cfg: dict) -> dict[str, int]:
    if cfg["mask_source"] == "totalsegmentator":
        from totalsegmentator.map_to_binary import class_map
        return {v: k for k, v in class_map[cfg["ts_task"]].items()}
    lm = cfg.get("custom_label_names")  # optional json {id: name}
    return {v: int(k) for k, v in json.load(open(lm)).items()} if lm else {}


def preprocess(cfg, cid, plans, lookup, img=None, msk=None):
    """img/msk may be passed in already loaded (worker.py) to avoid re-reading the .nii.gz."""
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    from nibabel.processing import resample_from_to

    if img is None:
        img = nib.load(str(Path(cfg["ct_dir"]) / f"{cid}.nii.gz"))
    img = as_closest_canonical(img)
    img = change_spacing(img, plans["spacing"], order=1, dtype=np.int32, nr_cpus=4)
    if msk is None:
        mpath = (cfg["out"] / "masks" / f"{cid}.nii.gz") if cfg["mask_source"] == "totalsegmentator" \
            else Path(cfg["mask_dir"]) / f"{cid}.nii.gz"
        msk = nib.load(str(mpath))
    msk = as_closest_canonical(msk)
    regrid = msk.shape != img.shape or not np.allclose(msk.affine, img.affine, atol=1e-3)
    if regrid:
        msk = resample_from_to(msk, img, order=0)
    lab = np.asanyarray(msk.dataobj).astype(np.int32).transpose(2, 1, 0)
    hu = np.asanyarray(img.dataobj).astype(np.int16).transpose(2, 1, 0)
    return hu, units_from_labels(cfg, lab, lookup), img.affine, regrid


def resample_ct(img, plans):
    """TS-identical image preprocessing: canonical (RAS) + change_spacing(order=1, int32).
    Returns (resampled nibabel image, hu array in nnU-Net (z,y,x) order)."""
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    img = change_spacing(as_closest_canonical(img), plans["spacing"], order=1, dtype=np.int32, nr_cpus=4)
    return img, np.asanyarray(img.dataobj).astype(np.int16).transpose(2, 1, 0)


def units_from_labels(cfg, lab, lookup):
    units = np.zeros(lab.shape, np.uint8)
    for k, (u, classes) in enumerate(cfg["organ_map"].items()):
        ids = [c if isinstance(c, int) else lookup[c] for c in classes]
        units[np.isin(lab, ids)] = k + 1
    return units


def tiles_for(lo, hi, cen, P, overlap):
    """per-axis start positions; organ extent <= P -> one patch at centroid (shifted to contain bbox)."""
    starts = []
    for a in range(3):
        ext = hi[a] - lo[a] + 1
        if ext <= P[a]:
            s = int(round(cen[a])) - P[a] // 2
            s = min(max(s, int(hi[a]) - P[a] + 1), int(lo[a]))
            starts.append([int(s)])
        else:
            stride = P[a] * (1 - overlap)
            n = math.ceil((ext - P[a]) / stride) + 1
            starts.append([int(round(x)) for x in np.linspace(lo[a], hi[a] - P[a] + 1, n)])
    return [list(s) for s in itertools.product(*starts)]


def process(cfg, cid, plans, lookup, img=None, msk=None):
    hu, units, affine, regrid = preprocess(cfg, cid, plans, lookup, img, msk)
    return finalize(cfg, cid, hu, units, affine, regrid, plans)


def finalize(cfg, cid, hu, units, affine, regrid, plans):
    """unit stats + patch tiles; writes prep/{id}.npz then patches/{id}.json (the done marker)."""
    P, names = plans["patch_size"], unit_names(cfg)
    info = {"id": cid, "shape": list(hu.shape), "patch_size": P, "spacing": plans["spacing"],
            "mask_regridded": bool(regrid), "units": {}}
    for k, u in enumerate(names):
        idx = np.argwhere(units == k + 1)
        nvox = int(len(idx))
        d = {"voxels": nvox, "present": nvox >= cfg["min_organ_voxels"]}
        if nvox:
            lo, hi, cen = idx.min(0), idx.max(0), idx.mean(0)
            d.update(bbox=[lo.tolist(), hi.tolist()], centroid=[round(float(c), 2) for c in cen])
        if d["present"]:
            tl = []
            for s in tiles_for(lo, hi, cen, P, cfg["tile_overlap"]):
                sl = tuple(slice(max(s[a], 0), s[a] + P[a]) for a in range(3))
                n_in = int((units[sl] == k + 1).sum())
                if n_in:
                    tl.append({"start": s, "n_vox": n_in})
            d["tiles"] = tl
        info["units"][u] = d
    np.savez_compressed(cfg["out"] / "prep" / f"{cid}.npz", hu=hu, units=units, affine=affine,
                        unit_names=np.array(names))
    (cfg["out"] / "patches" / f"{cid}.json").write_text(json.dumps(info))
    return info


def write_stats(cfg):
    rows = []
    for p in sorted((cfg["out"] / "patches").glob("*.json")):
        info = json.loads(p.read_text())
        for u, d in info["units"].items():
            rows.append({"id": info["id"], "unit": u, "voxels": d["voxels"], "present": d["present"],
                         "n_tiles": len(d.get("tiles", []))})
    df = pd.DataFrame(rows)
    df.to_csv(cfg["out"] / "mask_stats_per_case.csv", index=False)
    s = df.groupby("unit", sort=False).agg(n_cases=("id", "count"), present_rate=("present", "mean"),
                                           median_voxels=("voxels", "median"), min_voxels=("voxels", "min"),
                                           max_tiles=("n_tiles", "max"), mean_tiles=("n_tiles", "mean"))
    s.to_csv(cfg["out"] / "mask_stats.csv")
    print(s.to_string())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--stats", action="store_true", help="only rebuild mask_stats.csv")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.stats:
        write_stats(cfg)
        return
    plans, lookup = read_plans(cfg), label_lookup(cfg)
    ids = shard(case_ids(cfg), a.shard)
    todo = [i for i in ids if not (cfg["out"] / "patches" / f"{i}.json").exists()]
    mdir = cfg["out"] / "masks" if cfg["mask_source"] == "totalsegmentator" else Path(cfg["mask_dir"])
    todo = [i for i in todo if (mdir / f"{i}.nii.gz").exists()]
    t0, n = time.time(), 0
    for k, cid in enumerate(todo):
        t = time.time()
        try:
            info = process(cfg, cid, plans, lookup)
            n += 1
            nt = sum(len(d.get("tiles", [])) for d in info["units"].values())
            print(f"[{k+1}/{len(todo)}] {cid} shape={info['shape']} tiles={nt} {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            with open(cfg["out"] / "patches" / "failed.jsonl", "a") as f:
                f.write(json.dumps({"id": cid, "err": repr(e)}) + "\n")
            print(cid, "FAILED", repr(e), flush=True)
    write_stats(cfg)
    log_progress(f"patches.py shard={a.shard}: {n}/{len(todo)} cases preprocessed+patched in {time.time()-t0:.0f}s; "
                 f"stats -> {cfg['out']/'mask_stats.csv'}")


if __name__ == "__main__":
    main()
