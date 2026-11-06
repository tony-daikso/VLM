"""Step 1a: organ masks on raw HU.

mask_source=totalsegmentator: run TotalSegmentator (task/fast from config) with save_lowres=True so the
multilabel mask is written on the model grid (3mm for --fast). Cached in outputs/masks/{id}.nii.gz.
mask_source=custom: nothing to do here; patches.py reads mask_dir/{id}.nii.gz directly.
Resumable: existing masks are skipped; failures go to outputs/masks/failed.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import case_ids, load_cfg, log_progress, shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if cfg["mask_source"] != "totalsegmentator":
        print("mask_source != totalsegmentator; masks are read from mask_dir in patches.py")
        return
    from totalsegmentator.python_api import totalsegmentator

    out = cfg["out"] / "masks"
    ids = shard(case_ids(cfg), a.shard)
    todo = [i for i in ids if not (out / f"{i}.nii.gz").exists()]
    print(f"{len(ids)} cases, {len(todo)} to do")
    n_ok, t0 = 0, time.time()
    for k, cid in enumerate(todo):
        t = time.time()
        tmp = out / f"{cid}.tmp.nii.gz"
        try:
            totalsegmentator(str(Path(cfg["ct_dir"]) / f"{cid}.nii.gz"), str(tmp), ml=True,
                             fast=bool(cfg.get("ts_fast")), task=cfg["ts_task"], save_lowres=bool(cfg.get("ts_fast")),
                             device="gpu" if cfg["device"].startswith("cuda") else "cpu",
                             quiet=True, nr_thr_resamp=4, nr_thr_saving=2)
            tmp.rename(out / f"{cid}.nii.gz")
            n_ok += 1
            print(f"[{k+1}/{len(todo)}] {cid} {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            with open(out / "failed.jsonl", "a") as f:
                f.write(json.dumps({"id": cid, "err": repr(e)}) + "\n")
            print(f"[{k+1}/{len(todo)}] {cid} FAILED {e!r}", flush=True)
    log_progress(f"masks.py shard={a.shard}: {n_ok}/{len(todo)} new TS masks in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
