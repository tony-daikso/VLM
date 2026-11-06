"""Bone-unit embeddings for the fracture study (frozen encoder, same pipeline as extract.py).

Per case with masks/{id}.nii.gz + prep/{id}.npz (+ patches/{id}.json as the "prep complete" marker):
  117-class TS mask -> bone units (organ_map_bones.yaml; >= min_organ_voxels -> present)
  -> tiles (patches.tiles_for, tile_overlap) -> windows baseline / soft_tissue / bone -> frozen encoder
  -> per stage in stages_used:
       emb     : mask-weighted mean (extract.pool_stage, incl. bbox fallback), per-stage L2 norm, concat,
                 tiles averaged weighted by organ voxels in the tile (identical to extract.embed_case)
       emb_max : max over stage voxels with mask fraction >= max_frac (over all tiles of the unit);
                 if no such voxel in any tile -> the (tile-weighted) raw mean of that stage; per-stage L2 norm, concat
Output: {out}/<--emb-subdir, default emb_bones>/<window_key>/{id}.npz  emb [U,D], emb_max [U,D] (float16), organ_present, organ_names,
        n_tiles, n_vox, max_fallback [U,S], n_fallback [U,S], stages_used, stage_dims.
Resumable (skips cases whose files exist for all windows), atomic writes, per-case timing in {out}/extract_bones_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import ROOT, load_cfg, shard
from encoder import build_encoder
from extract import crop, pool_stage, window_key
from patches import label_lookup, tiles_for, units_from_labels
from windowing import apply_window

WINDOWS = ["baseline", "soft_tissue", "bone"]


def load_units(cfg, cid, lookup):
    """unit label volume (z,y,x) uint8 aligned with prep hu."""
    prep = np.load(cfg["out"] / "prep" / f"{cid}.npz")
    hu, aff = prep["hu"], prep["affine"]
    msk = nib.load(str(cfg["out"] / "masks" / f"{cid}.nii.gz"))
    regrid = msk.shape != hu.shape[::-1] or not np.allclose(msk.affine, aff, atol=1e-3)
    if regrid:  # same handling as patches.preprocess
        from nibabel.processing import resample_from_to
        from totalsegmentator.alignment import as_closest_canonical
        msk = resample_from_to(as_closest_canonical(msk), (hu.shape[::-1], aff), order=0)
    lab = np.asanyarray(msk.dataobj).astype(np.int32).transpose(2, 1, 0)
    return hu, units_from_labels(cfg, lab, lookup), regrid


def unit_tiles(cfg, units, U, P):
    """per unit: list of (start, n_vox) like patches.finalize; [] if not present."""
    out, nvox = [], np.zeros(U, np.int64)
    for k in range(U):
        idx = np.argwhere(units == k + 1)
        nvox[k] = len(idx)
        tl = []
        if len(idx) >= cfg["min_organ_voxels"]:
            lo, hi, cen = idx.min(0), idx.max(0), idx.mean(0)
            for s in tiles_for(lo, hi, cen, P, cfg["tile_overlap"]):
                sl = tuple(slice(max(s[a], 0), s[a] + P[a]) for a in range(3))
                n_in = int((units[sl] == k + 1).sum())
                if n_in:
                    tl.append((s, n_in))
        out.append(tl)
    return out, nvox


def max_stage(feat, mask, thr):
    """feat [W,C,z,y,x], mask [1,1,Z,Y,X] -> [W,C] max over stage voxels with mask fraction >= thr, or None."""
    f = [m // s for m, s in zip(mask.shape[2:], feat.shape[2:])]
    sel = F.avg_pool3d(mask, kernel_size=f, stride=f)[0, 0] >= thr
    if not bool(sel.any()):
        return None
    return feat.float()[:, :, sel].amax(2)


@torch.no_grad()
def embed_bones(cfg, enc, hu, units, wins, U, max_frac):
    P, stages = list(enc.patch_size), cfg["stages_used"]
    W, S = len(wins), len(stages)
    dims = enc.stage_channels(stages)
    tiles, nvox = unit_tiles(cfg, units, U, P)
    emb = np.zeros((U, W, sum(dims)), np.float32)
    emb_max = np.zeros((U, W, sum(dims)), np.float32)
    n_fallback = np.zeros((U, S), np.int32)
    max_fb = np.zeros((U, S), bool)
    for k in range(U):
        if not tiles[k]:
            continue
        acc, wsum = 0.0, 0
        raw_mean = [0.0] * S          # tile-weighted raw stage mean (fallback for max)
        mx = [None] * S
        for start, n_in in tiles[k]:
            h = torch.from_numpy(crop(hu, start, P, cfg["pad_hu"])).to(enc.device)
            m = torch.from_numpy(crop(units, start, P, 0) == k + 1).float().to(enc.device)[None, None]
            x = torch.stack([apply_window(h, w, enc.norm) for _, w in wins])[:, None]
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
                feats = enc.stage_features(x, upto=max(stages))
            vs = []
            for j, s in enumerate(stages):
                v, fb, _ = pool_stage(feats[s], m, cfg["min_stage_weight"])
                n_fallback[k, j] += fb
                vs.append(F.normalize(v, dim=1))
                raw_mean[j] = raw_mean[j] + v * n_in
                t = max_stage(feats[s], m, max_frac)
                if t is not None:
                    mx[j] = t if mx[j] is None else torch.maximum(mx[j], t)
            acc = acc + torch.cat(vs, 1) * n_in
            wsum += n_in
        emb[k] = (acc / wsum).cpu().numpy()
        parts = []
        for j in range(S):
            if mx[j] is None:
                max_fb[k, j] = True
                mx[j] = raw_mean[j] / wsum
            parts.append(F.normalize(mx[j], dim=1))
        emb_max[k] = torch.cat(parts, 1).cpu().numpy()
    present = np.array([bool(t) for t in tiles])
    n_tiles = np.array([len(t) for t in tiles], np.int32)
    names = np.array(list(cfg["organ_map"].keys()))
    out = {}
    for i, (wk, _) in enumerate(wins):
        out[wk] = dict(emb=emb[:, i].astype(np.float16), emb_max=emb_max[:, i].astype(np.float16),
                       organ_present=present, organ_names=names, n_tiles=n_tiles, n_vox=nvox,
                       n_fallback=n_fallback, max_fallback=max_fb, max_frac=np.float32(max_frac),
                       stages_used=np.array(stages), stage_dims=np.array(dims))
    return out


def out_path(cfg, wk, cid):
    return cfg["out"] / cfg.get("emb_subdir", "emb_bones") / wk / f"{cid}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--organ-map", default="organ_map_bones.yaml")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--ids-file", default=None, help="only these case ids (one per line)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N cases (smoke test)")
    ap.add_argument("--max-frac", type=float, default=0.5)
    ap.add_argument("--emb-subdir", default="emb_bones", help="output subdir under output_dir (Exp G: emb_ribs)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    cfg["emb_subdir"] = a.emb_subdir
    cfg["organ_map"] = yaml.safe_load(open(ROOT / a.organ_map))
    U = len(cfg["organ_map"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    lookup = label_lookup(cfg)
    wins = [(window_key(n, cfg["windows"][n]), cfg["windows"][n]) for n in WINDOWS]
    for wk, _ in wins:
        (cfg["out"] / a.emb_subdir / wk).mkdir(parents=True, exist_ok=True)
    out = cfg["out"]
    ids = sorted(p.name[:-7] for p in (out / "masks").glob("*.nii.gz"))
    ids = [i for i in ids if (out / "prep" / f"{i}.npz").exists() and (out / "patches" / f"{i}.json").exists()]
    if a.ids_file:
        keep = {l.strip() for l in open(a.ids_file) if l.strip()}
        ids = [i for i in ids if i in keep]
    ids = shard(ids, a.shard)
    todo = [i for i in ids if not all(out_path(cfg, wk, i).exists() for wk, _ in wins)]
    if a.limit:
        todo = todo[:a.limit]
    enc = build_encoder(cfg)
    print(f"{socket.gethostname()} shard={a.shard}: {len(ids)} cases, {len(todo)} to do, units={U}", flush=True)
    logf = out / ("extract_bones_log.csv" if a.emb_subdir == "emb_bones" else f"extract_{a.emb_subdir}_log.csv")
    new = not logf.exists()
    t0, done, nerr = time.time(), 0, 0
    with open(logf, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["id", "shard", "seconds", "peak_gpu_gib", "n_present", "n_tiles", "regrid", "time"])
        for k, cid in enumerate(todo):
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                hu, units, regrid = load_units(cfg, cid, lookup)
                res = embed_bones(cfg, enc, hu, units, wins, U, a.max_frac)
            except Exception as e:  # noqa: BLE001
                nerr += 1
                print(cid, "ERROR", repr(e), flush=True)
                with open(out / a.emb_subdir / "failed.jsonl", "a") as f:
                    f.write(json.dumps({"id": cid, "err": repr(e)}) + "\n")
                continue
            for wk, arrs in res.items():
                dst = out_path(cfg, wk, cid)
                tmp = dst.with_name(f"{cid}.{os.getpid()}.tmp.npz")
                np.savez(tmp, **arrs)
                tmp.rename(dst)
            dt, pk = time.time() - t, torch.cuda.max_memory_allocated() / 2**30
            a0 = next(iter(res.values()))
            wr.writerow([cid, a.shard, f"{dt:.2f}", f"{pk:.2f}", int(a0["organ_present"].sum()),
                         int(a0["n_tiles"].sum()), int(regrid), time.strftime("%H:%M:%S")])
            fh.flush()
            done += 1
            if k % 50 == 0 or k < 5:
                el = time.time() - t0
                print(f"[{k+1}/{len(todo)}] {cid} {dt:.2f}s present={int(a0['organ_present'].sum())}/{U} "
                      f"tiles={int(a0['n_tiles'].sum())} avg {el/(k+1):.2f}s/case", flush=True)
    print(f"DONE shard={a.shard}: {done} embedded, {nerr} errors, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
