"""Experiment L: 1.5 mm per-rib encoder (TotalSegmentator high-res rib model, nnU-Net Dataset295 = TS 'total'
part5_ribs, 1.5 mm, patch 128^3) on train+val cases. Exp J (vert15_full.py) recipe applied to the ribs.

per case:
  CT (ssh cat from CT-rate HDD, or local file) -> canonical -> ROI = bbox of the 3 mm Dataset297 rib labels
  (+-MARGIN mm per axis) -> change_spacing 1.5 -> Dataset295 seg (seg_nnunet) -> 18 rib units (organ_map_ribs.yaml:
  ribs 1-4 merged per side, ribs 5-12 single, per side) -> pilot_vert15.embed_units with the Dataset295 encoder
  (windows baseline / soft_tissue / bone, stages 1-3, mean (emb) + max (emb_max) pooling).
Outputs (out dir):
  emb15/{id}.npz : emb, emb_max [18,W,448] f16, present (nvox >= MIN_VOX), nvox, nfb, units, windows, stages, shape15
  seg15/{id}.npz : lab (uint8 Dataset295 labels 1-26 cropped to their bbox, (z,y,x) 1.5 mm grid), off, affine,
                   shape15, labels json   (compact rib masks; no HU)
  claims/{id}, timing_w{wid}.csv, failed.txt, logs/
Sub-commands:
  prio    : ordered train+val id list. tier0 = all train fracture positives (rib positives first) + equal number of
            train negatives (half hard: age>=75 / osteoporosis / metal-hardware, half random) + all val; tier1 = rest.
            Labels come from bones/expI/meta.csv (built from train/val rows only); test ids never touched.
  extract : worker (run_ribs15.sh recycles it)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from vert15_full import claim, fetch  # noqa: E402

RES = Path.home() / ".totalsegmentator/nnunet/results"
M15 = RES / "Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
M3 = RES / "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
UNITS = ([("rib_left_1_4", [f"rib_left_{i}" for i in range(1, 5)])] + [(f"rib_left_{i}", [f"rib_left_{i}"]) for i in range(5, 13)]
         + [("rib_right_1_4", [f"rib_right_{i}" for i in range(1, 5)])] + [(f"rib_right_{i}", [f"rib_right_{i}"]) for i in range(5, 13)])
MARGIN = 15.0   # mm around the 3 mm rib bbox
MIN_VOX = 200   # 1.5 mm voxels (= 25 voxels at 3 mm); stricter thresholds can be applied later from nvox


def cmd_prio(a):
    import pandas as pd
    spl = pd.read_csv(a.split).set_index("study_id")["split"]
    trval = spl[spl.isin(["train", "val"])]
    m = pd.read_csv(a.meta, index_col=0, keep_default_na=False, na_values=[""])
    assert set(m.split) <= {"train", "val"}
    m = m[m.index.isin(trval.index)]
    for c in ("site_rib", "hn_age75", "hn_osteo", "hn_hardware", "hn_metal"):
        m[c] = m[c].astype(str).eq("True")
    df = pd.DataFrame({"study_id": trval.index, "split": trval.values})
    df["y"] = df.study_id.map(m["y"])
    df["rib"] = df.study_id.map(m["site_rib"]).fillna(False).astype(bool)
    hard = (m.hn_age75 | m.hn_osteo | m.hn_hardware | m.hn_metal)
    df["hard"] = df.study_id.map(hard).fillna(False).astype(bool)
    rng = np.random.default_rng(0)
    tr = df[df.split == "train"]
    pos = tr[tr.y == 1]
    negs = tr[tr.y == 0]
    nh = len(pos) // 2
    hn = np.sort(negs[negs.hard].study_id.values)
    pick_h = rng.choice(hn, min(nh, len(hn)), replace=False)
    rest = np.sort(negs[~negs.study_id.isin(pick_h)].study_id.values)
    pick_r = rng.choice(rest, len(pos) - len(pick_h), replace=False)
    df["group"] = np.where(df.y == 1, np.where(df.rib, "pos_rib", "pos_other"),
                           np.where(df.y == 0, "neg", "nolabel"))
    df.loc[df.study_id.isin(pick_h), "group"] = "neg_hard"
    t0 = set(pos.study_id) | set(pick_h) | set(pick_r) | set(df[df.split == "val"].study_id)
    df["tier"] = np.where(df.study_id.isin(t0), 0, 1)
    # inside tier0: train rib positives first, then a shuffled mix of everything else
    df["pri"] = np.where((df.tier == 0) & (df.group == "pos_rib") & (df.split == "train"), 0, np.where(df.tier == 0, 1, 2))
    df = df.iloc[rng.permutation(len(df))].sort_values("pri", kind="stable").drop(columns="pri")
    assert not (df.split == "test").any()
    df.to_csv(a.out, index=False)
    print(df.groupby(["tier", "split", "group"]).size().to_string())
    df.study_id.to_csv(Path(a.out).with_suffix(".txt"), index=False, header=False)


def rib_world_corners(img3, lab3, names3):
    """8 world (RAS mm) corners of the bbox of the 3 mm rib labels, or None."""
    ids = [v for k, v in names3.items() if k.startswith("rib_")]
    idx = np.argwhere(np.isin(lab3, ids))  # (z,y,x) on the 3 mm grid
    if len(idx) < 50:
        return None
    lo, hi = idx.min(0)[::-1], idx.max(0)[::-1]
    corners = np.array([[a, b, c, 1] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], float)
    return (img3.affine @ corners.T)[:3].T


def rib_roi(canon, wc):
    vox = (np.linalg.inv(canon.affine) @ np.c_[wc, np.ones(len(wc))].T)[:3].T
    vlo, vhi = vox.min(0), vox.max(0)
    zooms = np.array(canon.header.get_zooms()[:3], float)
    sl = []
    for ax in range(3):
        a0 = int(np.floor(vlo[ax] - (MARGIN + 3) / zooms[ax]))
        a1 = int(np.ceil(vhi[ax] + (MARGIN + 3) / zooms[ax])) + 1
        sl.append(slice(max(a0, 0), min(a1, canon.shape[ax])))
    return canon.slicer[tuple(sl)]


def cmd_boxes(a):
    """precompute the world rib corners for a list of ids (small file shipped to remote workers instead of masks)."""
    import nibabel as nib
    from concurrent.futures import ProcessPoolExecutor
    names3 = json.loads((M3 / "dataset.json").read_text())["labels"]
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    dirs = [Path(d) for d in a.mask_dirs]

    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_box_job, [(cid, [str(d) for d in dirs]) for cid in ids], chunksize=20))
    ok = [i for i, r in enumerate(res) if r is not None]
    np.savez(a.out, ids=np.array([ids[i] for i in ok]), corners=np.stack([res[i] for i in ok]))
    print(f"boxes: {len(ok)}/{len(ids)}", flush=True)


def _box_job(arg):
    import nibabel as nib
    cid, dirs = arg
    names3 = json.loads((M3 / "dataset.json").read_text())["labels"]
    mp = next((Path(d) / f"{cid}.nii.gz" for d in dirs if (Path(d) / f"{cid}.nii.gz").exists()), None)
    if mp is None:
        return None
    m3 = nib.load(str(mp))
    return rib_world_corners(m3, np.asanyarray(m3.dataobj).astype(np.int32).transpose(2, 1, 0), names3)


def units_from(lab, enc):
    units = np.zeros(lab.shape, np.uint8)
    for k, (_, parts) in enumerate(UNITS):
        units[np.isin(lab, [enc.labels[p] for p in parts])] = k + 1
    return units


def save_seg(out: Path, cid, lab15, aff15, enc):
    m = (lab15 > 0)
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    tmp = out / "seg15" / f"{cid}.tmp.npz"
    np.savez_compressed(tmp, lab=lab15[sl].astype(np.uint8), off=lo, affine=aff15, labels=json.dumps(enc.labels),
                        shape15=np.array(lab15.shape))
    tmp.rename(out / "seg15" / f"{cid}.npz")


def cmd_extract(a):
    import nibabel as nib
    import torch
    import pilot_vert15 as P
    from encoder import NnUNetEncoder
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    from worker import seg_nnunet, with_oom_retry

    if a.windows != "all":
        keep = a.windows.split(",")
        P.WINDOWS = [w for w in P.WINDOWS if w[0] in keep]
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = False
    out = Path(a.out)
    for d in ("emb15", "seg15", "claims", "ct"):
        (out / d).mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    mask_dirs = [Path(d) for d in (a.mask_dirs or [])]
    boxes = {}
    if a.boxes:
        zb = np.load(a.boxes)
        boxes = {str(i): c for i, c in zip(zb["ids"], zb["corners"])}
    names3 = json.loads((M3 / "dataset.json").read_text())["labels"]
    enc = NnUNetEncoder(M15)
    logf = out / f"timing_w{a.wid}.csv"
    new = not logf.exists()
    fh = open(logf, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["id", "pull", "load", "roi_resample15", "seg15", "embed15", "save", "total_proc", "shape15",
                     "n15", "status", "peak_mb"])

    def next_ids():
        if not a.consume:
            for cid in ids:
                if (out / "STOP").exists():
                    return
                if (out / "emb15" / f"{cid}.npz").exists():
                    continue
                if claim(out, cid):
                    yield cid
            return
        buf = Path(a.ct_local)          # stream mode: take whatever the downloader has finished, in list order
        while not (out / "STOP").exists():
            have = {p.name[:-7] for p in buf.glob("*.nii.gz") if not p.name.startswith(".")}
            got = False
            for cid in ids:
                if cid in have and not (out / "emb15" / f"{cid}.npz").exists() and claim(out, cid):
                    got = True
                    yield cid
                    break
            if not got:
                if (buf / "DONE_DOWNLOAD").exists() and not have - {p.name for p in (out / "claims").iterdir()}:
                    return
                time.sleep(3)

    gen = next_ids()
    pre = {}

    def prefetch(cid):
        t = time.time()
        try:
            pre[cid] = fetch(cid, out / "ct" / f"{cid}.nii.gz", a.ct_local)
        except Exception as e:  # noqa: BLE001
            pre[cid] = e
        pre[cid + "_t"] = time.time() - t

    def start_next():
        cid = next(gen, None)
        if cid is None:
            return None, None
        th = threading.Thread(target=prefetch, args=(cid,))
        th.start()
        return cid, th

    lim = a.limit or a.max_cases
    n = 0
    cid, th = start_next()
    while cid is not None and (not lim or n < lim):
        th.join()
        ncid, nth = start_next() if (not lim or n + 1 < lim) else (None, None)
        T = {"pull": pre.get(cid + "_t", np.nan)}
        ct = pre.get(cid)
        torch.cuda.reset_peak_memory_stats()
        try:
            if isinstance(ct, Exception) or ct is None:
                raise RuntimeError(f"pull failed: {ct!r}")
            t0 = t = time.time()
            wc = boxes.get(cid)
            if wc is None:
                mp = next((d / f"{cid}.nii.gz" for d in mask_dirs if (d / f"{cid}.nii.gz").exists()), None)
                if mp is None:
                    raise RuntimeError("no 3 mm mask / box")
                m3 = nib.load(str(mp))
                wc = rib_world_corners(m3, np.asanyarray(m3.dataobj).astype(np.int32).transpose(2, 1, 0), names3)
            raw = nib.load(str(ct))
            img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header)
            img.header.set_slope_inter(1, 0)
            canon = as_closest_canonical(img)
            T["load"] = time.time() - t; t = time.time()
            if wc is None:
                raise RuntimeError("no ribs in 3 mm mask")
            roi = rib_roi(canon, wc)
            if roi is None:
                raise RuntimeError("no ribs in 3 mm mask")
            img15 = change_spacing(roi, list(enc.spacing), order=1, dtype=np.int32, nr_cpus=a.threads)
            hu15 = np.asanyarray(img15.dataobj).astype(np.int16).transpose(2, 1, 0)
            del img, canon, raw
            T["roi_resample15"] = time.time() - t; t = time.time()
            lab15 = with_oom_retry(lambda: seg_nnunet(enc, hu15))
            T["seg15"] = time.time() - t; t = time.time()
            units = units_from(lab15, enc)
            r = P.embed_units(enc, hu15, units, len(UNITS), MIN_VOX)
            T["embed15"] = time.time() - t; t = time.time()
            if (lab15 > 0).any():
                save_seg(out, cid, lab15, img15.affine, enc)
            tmp = out / "emb15" / f"{cid}.tmp.npz"
            np.savez(tmp, units=np.array([u for u, _ in UNITS]), windows=np.array([w for w, _ in P.WINDOWS]),
                     stages=np.array(P.STAGES), shape15=np.array(hu15.shape), **r)
            tmp.rename(out / "emb15" / f"{cid}.npz")
            T["save"] = time.time() - t
            T["total_proc"] = time.time() - t0
            status, s15, n15 = "ok", "x".join(map(str, hu15.shape)), int(r["present"].sum())
        except Exception as e:  # noqa: BLE001
            status, s15, n15 = f"FAIL {e!r}"[:200].replace("\n", " "), "", 0
            with open(out / "failed.txt", "a") as ff:
                ff.write(f"{cid}\t{status}\n")
        finally:
            if (not a.ct_local or a.consume) and isinstance(ct, Path):
                ct.unlink(missing_ok=True)
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 2**20
        wr.writerow([cid] + [f"{T.get(k, np.nan):.2f}" for k in ["pull", "load", "roi_resample15", "seg15",
                     "embed15", "save", "total_proc"]] + [s15, n15, status, f"{peak:.0f}"])
        fh.flush()
        print(f"w{a.wid} [{n+1}] {cid} {status} n15={n15} shape={s15} peak={peak:.0f}MB " +
              " ".join(f"{k}={v:.1f}" for k, v in T.items()), flush=True)
        n += 1
        cid, th = ncid, nth
    if cid is not None and th is not None:  # prefetched but not processed: release the claim
        th.join()
        ct = pre.get(cid)
        if not a.ct_local and isinstance(ct, Path):
            ct.unlink(missing_ok=True)
        (out / "claims" / cid).unlink(missing_ok=True)
    fh.close()
    if cid is None and n == 0 and not (out / "STOP").exists():
        (out / f"DONE_w{a.wid}").touch()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prio")
    p.add_argument("--split", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--ids", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--mask-dirs", nargs="*", default=[])
    e.add_argument("--boxes", default=None, help="npz from the boxes command (world rib corners) instead of masks")
    e.add_argument("--consume", action="store_true", help="stream mode: --ct-local is a download buffer; delete each CT")
    e.add_argument("--ct-local", default=None)
    e.add_argument("--wid", type=int, default=0)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--max-cases", type=int, default=0)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--windows", default="all")
    b = sp.add_parser("boxes")
    b.add_argument("--ids", required=True)
    b.add_argument("--mask-dirs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    {"prio": cmd_prio, "extract": cmd_extract, "boxes": cmd_boxes}[a.cmd](a)


if __name__ == "__main__":
    main()
