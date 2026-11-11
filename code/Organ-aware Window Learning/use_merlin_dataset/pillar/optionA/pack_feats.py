"""Option A, step 0 (CT-rate, numpy only, CPU, run with nice 19): pack the compact parts of the finished Pillar-0 npz files
into shards so they can be pulled to A6000 (grid16 / grid4 / meta are NOT copied).

Incremental: every run packs only the case ids that are not in an existing shard yet and writes one new shard
  <out>/shard_<k>.npz : ids (str), global (n,1152) f16, vec64 (n,118,384) f16, vec16 (n,118,384) f16,
                        w64 / w16 (n,118) f32, vox_total (n,118) i32, coverage (n,118) f32
Only finished files (<id>.npz, the atomic rename = done marker) are read; *.tmp.npz is ignored. The extractor only ever
writes train/val cases (test gated), and this script additionally keeps only ids whose split.csv split is train/val.
usage: OMP_NUM_THREADS=1 nice -n 19 python pack_feats.py --feats /media/daikso/Tony/pillar/merlin/feats \
           --out /media/daikso/Tony/pillar/merlin/packed [--max 0]
"""
import argparse
import csv
import glob
import os
import time

import numpy as np

SPLIT = "/media/daikso/Tony/OWL/data_h/split.csv"
KEYS = ("global", "vec64", "vec16", "w64", "w16", "vox_total", "coverage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0, help="pack at most this many new cases (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok = {r["study_id"] for r in csv.DictReader(open(SPLIT)) if r["split"] in ("train", "val")}
    done = set()
    shards = sorted(glob.glob(os.path.join(a.out, "shard_*.npz")))
    for s in shards:
        done |= set(np.load(s)["ids"].tolist())
    files = sorted(f for f in glob.glob(os.path.join(a.feats, "*.npz")) if not f.endswith(".tmp.npz"))
    new = [f for f in files if os.path.basename(f)[:-4] not in done and os.path.basename(f)[:-4] in ok]
    if a.max:
        new = new[:a.max]
    print(f"{len(files)} finished npz, {len(done)} already packed in {len(shards)} shards, {len(new)} new", flush=True)
    if not new:
        return
    n = len(new)
    buf = {"global": np.zeros((n, 1152), np.float16), "vec64": np.zeros((n, 118, 384), np.float16),
           "vec16": np.zeros((n, 118, 384), np.float16), "w64": np.zeros((n, 118), np.float32),
           "w16": np.zeros((n, 118), np.float32), "vox_total": np.zeros((n, 118), np.int32),
           "coverage": np.full((n, 118), np.nan, np.float32)}
    ids, keep = [], []
    t0 = time.time()
    for i, f in enumerate(new):
        try:
            z = np.load(f)
            for k in KEYS:
                buf[k][i] = z[k]
            ids.append(os.path.basename(f)[:-4])
            keep.append(i)
        except Exception as e:  # noqa: BLE001  (file being replaced / truncated -> next run)
            print("skip", f, repr(e), flush=True)
        if i % 1000 == 0:
            print(f"  {i}/{n} {time.time() - t0:.0f}s", flush=True)
    keep = np.array(keep, int)
    k = len(shards)
    tmp = os.path.join(a.out, f"shard_{k:03d}.tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, ids=np.array(ids), **{kk: v[keep] for kk, v in buf.items()})
    os.replace(tmp, os.path.join(a.out, f"shard_{k:03d}.npz"))
    print(f"wrote shard_{k:03d} with {len(ids)} cases in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
