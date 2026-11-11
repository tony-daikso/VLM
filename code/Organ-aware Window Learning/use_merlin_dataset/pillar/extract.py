"""Resumable Pillar-0 feature extraction for Merlin (train -> val; test is gated behind --allow-test, off by default).

One GPU process (this one); CPU-only preprocessing workers (fork Pool created BEFORE CUDA init) with a bounded
prefetch window. Per case writes <out>/feats/<id>.npz atomically (tmp + rename = done marker); existing files are
skipped. CUDA OOM -> empty_cache + one retry, then the case is logged to failures.jsonl and skipped. A heartbeat
file is touched after every case for the external watchdog (run_extract.sh). A worker result that does not arrive
within --stall-timeout seconds makes the process exit(3) so the watchdog restarts it; a case that failed/stalled
--max-attempts times is skipped on later runs.

npz schema (all per case):
  global  (1152,) fp16   cat of per-scale token max-pool (scale 64^3, 16^3, 4^3), un-normalised
  grid16  (4096,384) fp16  stage-2 output tokens on the 16^3 grid (30 mm cells), raster (z,y,x) of the 384^3 input
  grid4   (64,384) fp16    final-stage tokens on the 4^3 grid (120 mm cells)
  vec64   (118,384) fp16   per TS-'total' label (index = TS label id, 0 unused) mask-weighted mean of 64^3 tokens
  vec16   (118,384) fp16   same on 16^3 tokens
  w64,w16 (118,) fp32      sum of token weights (label volume in cell units) -> composite unit of labels A,B:
                           (wA*vA + wB*vB)/(wA+wB) is exact
  vox_total,vox_crop (118,) int32  label voxels (1.25 mm grid) in the whole resampled scan / inside the 384^3 crop
  coverage (118,) fp32     vox_crop/vox_total (NaN if label absent)
  meta    json str         preprocessing plan, original size/spacing, mask source, versions
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pillar_common as pc  # noqa: E402

CT_DIR = Path("/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data")
SPLIT = Path("/media/daikso/Tony/OWL/data_h/split.csv")
MASK_DIRS = [Path("/media/daikso/Tony/OWL/mw_exp/outputs/masks"), Path("/media/daikso/Tony/pillar/masks_from_a6000")]
VERSION = "pillar0-abdct-v1 (rave 1.25mm, z384 centre, 11ch no-norm, autocast bf16)"


def find_mask(cid):
    for d in MASK_DIRS:
        p = d / f"{cid}.nii.gz"
        if p.exists():
            return str(p)
    return None


def case_list(splits):
    rows = list(csv.DictReader(open(SPLIT)))
    out = []
    for s in splits:
        out += sorted((r["study_id"], s) for r in rows if r["split"] == s)
    return out


def _init_worker(threads):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)


def _prep(args):
    cid, ct, mk = args
    try:
        return pc.prep_case(cid, ct, mk)
    except Exception as e:  # noqa: BLE001
        return dict(cid=cid, error=repr(e), tb=traceback.format_exc())


def gpu_case(visual, item, device="cuda"):
    import torch
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = pc.window_gpu(torch.from_numpy(item["vol"]).to(device))
        glob, grids = pc.visual_forward(visual, x)
        del x
        g64, g16, g4 = grids
        if item.get("mask_ok"):
            vec64, vec16, w64, w16 = pc.pool_units(g64, g16, item["cc64"], device)
        else:
            vec64 = vec16 = np.zeros((pc.N_LABELS, g64.shape[1]), np.float32)
            w64 = w16 = np.zeros(pc.N_LABELS)
        C = g16.shape[1]
        res = dict(
            glob=glob[0].float().cpu().numpy(),
            grid16=g16[0].reshape(C, -1).T.float().cpu().numpy(),
            grid4=g4[0].reshape(C, -1).T.float().cpu().numpy(),
            vec64=vec64, vec16=vec16, w64=w64, w16=w16,
        )
        del glob, grids, g64, g16, g4
    torch.cuda.synchronize()
    res["t_gpu"] = time.time() - t0
    return res


def save_case(out_dir, item, res, split, extra_meta=None):
    feats = out_dir / "feats"
    feats.mkdir(parents=True, exist_ok=True)
    vt = item.get("mask_total", np.zeros(pc.N_LABELS, np.int64))
    vc = item.get("mask_crop", np.zeros(pc.N_LABELS, np.int64))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(vt > 0, vc / np.maximum(vt, 1), np.nan).astype(np.float32)
    meta = dict(cid=item["cid"], split=split, version=VERSION, mask_ok=bool(item.get("mask_ok")),
                mask_path=item.get("mask_path"), **item["info"], **(extra_meta or {}))
    tmp = feats / f"{item['cid']}.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **{
            "global": res["glob"].astype(np.float16), "grid16": res["grid16"].astype(np.float16),
            "grid4": res["grid4"].astype(np.float16), "vec64": res["vec64"].astype(np.float16),
            "vec16": res["vec16"].astype(np.float16), "w64": np.asarray(res["w64"], np.float32),
            "w16": np.asarray(res["w16"], np.float32), "vox_total": vt.astype(np.int32),
            "vox_crop": vc.astype(np.int32), "coverage": cov, "meta": np.array(json.dumps(meta)),
        })
    os.replace(tmp, feats / f"{item['cid']}.npz")


def attempts(out_dir):
    p = out_dir / "failures.jsonl"
    n = {}
    if p.exists():
        for line in open(p):
            try:
                n[json.loads(line)["cid"]] = n.get(json.loads(line)["cid"], 0) + 1
            except Exception:  # noqa: BLE001
                pass
    return n


def log_fail(out_dir, cid, kind, msg):
    with open(out_dir / "failures.jsonl", "a") as f:
        f.write(json.dumps(dict(cid=cid, kind=kind, msg=msg[-2000:], t=time.strftime("%F %T"))) + "\n")


def run(cases, out_dir, workers=6, threads=2, prefetch=None, stall_timeout=900, max_attempts=2,
        heartbeat=None, on_case=None, visual=None, pool=None):
    """cases: list of (cid, split). Returns list of per-case timing dicts."""
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nfail = attempts(out_dir)
    todo = [(c, s) for c, s in cases if not (out_dir / "feats" / f"{c}.npz").exists() and nfail.get(c, 0) < max_attempts]
    print(f"[extract] {len(cases)} cases, {len(todo)} to do, workers={workers}", flush=True)
    if not todo:
        return []
    own_pool = pool is None
    if own_pool:
        pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(threads,))
    if visual is None:
        visual, miss, unexp = pc.load_model()
        print(f"[extract] model loaded; missing={len(miss)} unexpected={len(unexp)}", flush=True)
    prefetch = prefetch or workers + 2
    split_of = dict(todo)
    q = deque()
    it = iter(todo)

    def submit():
        try:
            c, _ = next(it)
        except StopIteration:
            return
        mk = find_mask(c)
        q.append((c, mk, pool.apply_async(_prep, ((c, str(CT_DIR / f"{c}.nii.gz"), mk),))))

    for _ in range(prefetch):
        submit()
    stats, t_start, done, consec_fail = [], time.time(), 0, 0
    while q:
        c, mk, ar = q.popleft()
        try:
            item = ar.get(timeout=stall_timeout)
        except mp.TimeoutError:
            log_fail(out_dir, c, "stall", f"prep > {stall_timeout}s")
            print(f"[extract] STALL on {c}; exiting for watchdog restart", flush=True)
            pool.terminate()
            sys.exit(3)
        submit()
        if "error" in item:
            log_fail(out_dir, c, "prep", item["tb"])
            print(f"[extract] prep FAIL {c}: {item['error']}", flush=True)
            continue
        item["mask_path"] = mk
        res = None
        for attempt in range(2):
            try:
                res = gpu_case(visual, item)
                break
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"[extract] OOM {c} attempt {attempt}", flush=True)
                err = repr(e)
            except Exception as e:  # noqa: BLE001
                err = traceback.format_exc()
                torch.cuda.empty_cache()
                break
        if res is None:
            consec_fail += 1
            if consec_fail >= 3:  # systemic GPU problem: do not burn attempts on every case
                print(f"[extract] 3 consecutive GPU failures, last {c}: {err[-300:]}; exit 5", flush=True)
                pool.terminate()
                sys.exit(5)
            log_fail(out_dir, c, "gpu", err)
            print(f"[extract] gpu FAIL {c}", flush=True)
            continue
        consec_fail = 0
        save_case(out_dir, item, res, split_of[c])
        done += 1
        st = dict(cid=c, t_ct=item["t_ct"], t_mask=item["t_mask"], t_gpu=res["t_gpu"],
                  peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        stats.append(st)
        if on_case:
            on_case(item, res, st)
        if heartbeat:
            Path(heartbeat).write_text(f"{time.time():.0f} {done} {c}\n")
        if done % 20 == 0 or done <= 3:
            el = time.time() - t_start
            print(f"[extract] {done}/{len(todo)} {el/done:.1f}s/case e2e | last ct {st['t_ct']:.1f}s mask "
                  f"{st['t_mask']:.1f}s gpu {st['t_gpu']:.2f}s peak {st['peak_gb']:.2f}GB | ETA "
                  f"{(len(todo)-done)*el/done/3600:.1f}h", flush=True)
    if own_pool:
        pool.close()
        pool.join()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/daikso/Tony/pillar/merlin")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stall-timeout", type=int, default=900)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--allow-test", action="store_true", help="also extract the TEST split (requires explicit user OK)")
    a = ap.parse_args()
    splits = ["train", "val"] + (["test"] if a.allow_test else [])
    cases = case_list(splits)
    run(cases, a.out, workers=a.workers, threads=a.threads, stall_timeout=a.stall_timeout, heartbeat=a.heartbeat)
    out = Path(a.out)
    nf = attempts(out)
    left = [c for c, _ in cases if not (out / "feats" / f"{c}.npz").exists() and nf.get(c, 0) < 2]
    if left:
        print(f"[extract] {len(left)} cases still pending", flush=True)
        sys.exit(4)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
