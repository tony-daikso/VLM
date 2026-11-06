"""Step 3: frozen-encoder multi-stage mask pooling -> organ embeddings [n_units, n_windows, D].

Per unit, per tile: crop HU (pad with pad_hu) -> every window -> frozen encoder (one batch of n_windows)
-> for each stage in stages_used: mask avg-pooled to the stage grid = fractional weights; weighted mean of
the feature map (bbox-mean fallback if total weight < min_stage_weight) -> per-stage L2 normalise -> concat.
Tiles of a unit are averaged weighted by organ voxels inside the tile.
Resumable: existing outputs/emb/{id}.npz are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import case_ids, load_cfg, log_progress, shard, unit_names
from encoder import build_encoder
from windowing import apply_window


def crop(arr: np.ndarray, start, P, pad):
    out = np.full(P, pad, dtype=arr.dtype)
    src = tuple(slice(max(s, 0), min(s + p, n)) for s, p, n in zip(start, P, arr.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = arr[src]
    return out


def pool_stage(feat: torch.Tensor, mask: torch.Tensor, min_w: float):
    """feat [W,C,z,y,x], mask [1,1,Z,Y,X] float -> ([W,C] pooled, used_fallback, raw_norm [W])"""
    f = [m // s for m, s in zip(mask.shape[2:], feat.shape[2:])]
    w = F.avg_pool3d(mask, kernel_size=f, stride=f)  # organ fraction per stage voxel
    fallback = bool(w.sum() < min_w)
    if fallback:
        nz = torch.nonzero(mask[0, 0])
        lo = (nz.min(0).values // torch.tensor(f, device=nz.device)).tolist()
        hi = (nz.max(0).values // torch.tensor(f, device=nz.device)).tolist()
        w = torch.zeros_like(w)
        w[..., lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1.0
    v = (feat.float() * w).sum((2, 3, 4)) / w.sum()
    return v, fallback, v.norm(dim=1)


def window_key(name: str, win: dict) -> str:
    """cache folder name; includes the HU parameters so editing a window in windows.yaml forces recompute."""
    if win.get("type") == "baseline":
        return "baseline"
    return f"{name}_L{win['level']:g}_W{win['width']:g}"


def emb_path(cfg, wkey, cid):
    return cfg["out"] / "emb" / wkey / f"{cid}.npz"


@torch.no_grad()
def embed_case(cfg, enc, cid, wins, units_list):
    """wins: list of (name, win dict) to compute. Returns {name: dict(arrays)} — one cache file per window."""
    prep = np.load(cfg["out"] / "prep" / f"{cid}.npz")
    hu, units = prep["hu"], prep["units"]
    info = json.loads((cfg["out"] / "patches" / f"{cid}.json").read_text())
    P, stages = list(enc.patch_size), cfg["stages_used"]
    W, U = len(wins), len(units_list)
    dims = enc.stage_channels(stages)
    emb = np.zeros((U, W, sum(dims)), np.float32)
    present = np.zeros(U, bool)
    raw_norm = np.zeros((U, W, len(stages)), np.float32)
    n_fallback = np.zeros((U, len(stages)), np.int32)
    n_tiles = np.zeros(U, np.int32)
    jobs = [(k, t) for k, u in enumerate(units_list) if info["units"][u]["present"] for t in info["units"][u]["tiles"]]
    acc = {}
    wsum = np.zeros(U)
    B = int(cfg.get("extract_tiles_per_batch", 4))  # tiles per forward; batch = B * n_windows patches
    for i0 in range(0, len(jobs), B):
        chunk = jobs[i0:i0 + B]
        xs, ms = [], []
        for k, t in chunk:
            h = torch.from_numpy(crop(hu, t["start"], P, cfg["pad_hu"])).to(enc.device)
            ms.append(torch.from_numpy(crop(units, t["start"], P, 0) == k + 1).float().to(enc.device)[None, None])
            xs += [apply_window(h, w, enc.norm) for _, w in wins]
        x = torch.stack(xs)[:, None]
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
            feats = enc.stage_features(x, upto=max(stages))
        for b, (k, t) in enumerate(chunk):
            vs = []
            for j, s in enumerate(stages):
                v, fb, rn = pool_stage(feats[s][b * W:(b + 1) * W], ms[b], cfg["min_stage_weight"])
                n_fallback[k, j] += fb
                raw_norm[k, :, j] += rn.cpu().numpy() * t["n_vox"]
                vs.append(F.normalize(v, dim=1))
            acc[k] = acc.get(k, 0.0) + torch.cat(vs, 1) * t["n_vox"]
            wsum[k] += t["n_vox"]
    for k, a in acc.items():
        emb[k] = (a / wsum[k]).cpu().numpy()
        raw_norm[k] /= wsum[k]
        present[k], n_tiles[k] = True, len(info["units"][units_list[k]]["tiles"])
    out = {}
    for i, (name, _) in enumerate(wins):
        out[name] = dict(emb=emb[:, i].astype(np.float16), organ_present=present, organ_names=np.array(units_list),
                         stages_used=np.array(stages), stage_dims=np.array(dims), raw_norm=raw_norm[:, i],
                         n_fallback=n_fallback, n_tiles=n_tiles)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=None, help="override output_dir for this host")
    ap.add_argument("--all-prep", action="store_true", help="process every case with prep/patches, not case_ids()")
    ap.add_argument("--watch", action="store_true", help="keep polling for new prep files until {out}/STOP_EXTRACT")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.out:
        cfg["out"] = Path(a.out)
    torch.manual_seed(cfg["seed"])
    # fixed patch shape -> autotuned conv3d kernels; off on small shared GPUs (big workspace)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    enc = build_encoder(cfg)
    ul = unit_names(cfg)
    wins = [(window_key(n, w), w) for n, w in cfg["windows"].items()]
    for k, _ in wins:
        (cfg["out"] / "emb" / k).mkdir(parents=True, exist_ok=True)
    logf = cfg["out"] / "extract_log.csv"
    new = not logf.exists()
    t0, done = time.time(), 0
    with open(logf, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["id", "n_windows", "seconds", "peak_gpu_gib", "n_tiles", "n_fallback"])
        while True:
            if a.all_prep or a.watch:
                ids = sorted(p.stem for p in (cfg["out"] / "patches").glob("*.json"))
            else:
                ids = case_ids(cfg)
            ids = shard(ids, a.shard)
            todo = []
            for cid in ids:
                miss = [(k, w) for k, w in wins if not emb_path(cfg, k, cid).exists()]
                if miss and (cfg["out"] / "patches" / f"{cid}.json").exists() and (cfg["out"] / "prep" / f"{cid}.npz").exists():
                    todo.append((cid, miss))
            if not todo:
                if a.watch and not (cfg["out"] / "STOP_EXTRACT").exists():
                    time.sleep(30)
                    continue
                break
            for k, (cid, miss) in enumerate(todo):
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                try:
                    out = embed_case(cfg, enc, cid, miss, ul)
                except Exception as e:  # noqa: BLE001  (e.g. prep still being written by a worker)
                    print(cid, "ERROR", repr(e), flush=True)
                    continue
                for wk, arrs in out.items():
                    tmp = emb_path(cfg, wk, cid).with_suffix(".tmp.npz")
                    np.savez(tmp, **arrs)
                    tmp.rename(emb_path(cfg, wk, cid))
                dt, pk = time.time() - t, torch.cuda.max_memory_allocated() / 2**30
                a0 = next(iter(out.values()))
                wr.writerow([cid, len(miss), f"{dt:.2f}", f"{pk:.2f}", int(a0["n_tiles"].sum()), int(a0["n_fallback"].sum())])
                fh.flush()
                done += 1
                if k % 50 == 0:
                    print(f"[{k+1}/{len(todo)}] {cid} windows={len(miss)} {dt:.1f}s peak {pk:.2f}GiB", flush=True)
            if not a.watch:
                break
    log_progress(f"extract.py shard={a.shard} out={cfg['out']}: {done} cases embedded in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
