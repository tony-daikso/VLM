"""Experiment J: 1.5 mm vertebra encoder (TotalSegmentator Dataset292) on all train+val cases.

Scale-up of pilot_vert15.py with the 3 mm seg skipped: the spine ROI comes from the existing Dataset297 3 mm masks
(outputs/masks/{id}.nii.gz, saved on the canonical 3 mm grid), so per case:
  CT (ssh cat from CT-rate, or local file) -> canonical -> ROI from the 3 mm vertebra labels (>= 210 mm x/y, +-100 mm z)
  -> change_spacing 1.5 -> Dataset292 seg (seg_nnunet) -> units T8..L5 -> embeddings (pilot_vert15.embed_units:
  windows baseline / soft_tissue / bone, stages 1-3, mean (emb) + max (emb_max) pooling).
Outputs (out dir):
  emb15/{id}.npz  : emb, emb_max [10,W,448] f16, present, nvox, nfb, levels, windows, stages, shape15
  seg15/{id}.npz  : lab (uint8, Dataset292 labels cropped to the vertebra bbox, (z,y,x) of the 1.5 mm grid),
                    off (crop offset z,y,x), affine (1.5 mm grid, RAS, (x,y,z) index order), labels json,
                    hu_slab (int16, HU in the same crop for x within body-centre +-slab voxels), slab_x0
  claims/{id}     : atomic claim (O_EXCL) so several workers/lives share one ordered id list
  timing_w{wid}.csv, failed.txt
Sub-commands:
  prio    : ordered id list (train+val only): tier0 = all train fracture positives + equal number of random train
            negatives + all val; tier1 = the rest of train. Split filter happens on the raw line BEFORE json parsing.
  extract : worker (see run_vert15.sh for the recycling loop)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

RES = Path.home() / ".totalsegmentator/nnunet/results"
M15 = RES / "Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
M3 = RES / "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
CT_HOST = "daikso@192.168.0.189"
CT_DIR = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
SLAB = 5  # hu_slab half-width in 1.5 mm voxels (+-7.5 mm around the median body x of T8..L5); HU outside the
          # vertebra labels dilated by 3 voxels is set to -2048 (compresses well)


# ----------------------------------------------------------------------------------------------- prio
def cmd_prio(a):
    import pandas as pd
    spl = pd.read_csv(a.split).set_index("study_id")["split"]
    trval = set(spl[spl.isin(["train", "val"])].index)
    rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"')
    grp = {}
    with open(a.labels) as fh:
        for line in fh:
            m = rx.search(line)
            if m is None or m.group(1) not in trval:   # test / unknown lines are never parsed
                continue
            r = json.loads(line)
            if r.get("split") not in ("train", "val") or spl.get(r["study_id"]) != r["split"]:
                continue
            fr = r["answer"].get("fracture", {})
            cat = fr.get("category")
            grp[r["study_id"]] = ("pos" if cat in ("yes", "probably_yes") else
                                  "neg" if cat in ("no", "probably_no", "not_mentioned") else "unclear")
    df = pd.DataFrame({"study_id": sorted(trval)})
    df["split"] = df.study_id.map(spl)
    df["group"] = df.study_id.map(grp).fillna("nolabel")
    assert not (df.split == "test").any()
    rng = np.random.default_rng(0)
    tr = df[df.split == "train"]
    pos = tr[tr.group == "pos"].study_id.values
    neg = np.sort(tr[tr.group == "neg"].study_id.values)
    negs = rng.choice(neg, len(pos), replace=False)
    t0 = set(pos) | set(negs) | set(df[df.split == "val"].study_id)
    df["tier"] = np.where(df.study_id.isin(t0), 0, 1)
    df = df.iloc[rng.permutation(len(df))].sort_values("tier", kind="stable")
    df.to_csv(a.out, index=False)
    print(df.groupby(["tier", "split", "group"]).size().to_string())
    df.study_id.to_csv(Path(a.out).with_suffix(".txt"), index=False, header=False)


# ----------------------------------------------------------------------------------------------- extract
def fetch(cid, dst: Path, local_dir: str | None):
    if local_dir:
        return Path(local_dir) / f"{cid}.nii.gz"
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["ssh", "-c", "aes128-gcm@openssh.com", "-o", "BatchMode=yes", CT_HOST,
                        f"cat {CT_DIR}/{cid}.nii.gz"], stdout=fh, check=True, timeout=900)
    tmp.rename(dst)
    return dst


def claim(out: Path, cid: str) -> bool:
    try:
        os.close(os.open(out / "claims" / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def save_seg(out: Path, cid, lab15, hu15, aff15, enc15, units):
    vids = [v for k, v in enc15.labels.items() if k.startswith("vertebrae_") or k == "sacrum"]
    m = np.isin(lab15, vids)
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    lab = np.where(m[sl], lab15[sl], 0).astype(np.uint8)
    body = np.argwhere(units[sl] > 0)
    xc = int(np.median(body[:, 2])) if len(body) else lab.shape[2] // 2
    x0, x1 = max(xc - SLAB, 0), min(xc + SLAB + 1, lab.shape[2])
    from scipy.ndimage import binary_dilation
    hu = hu15[sl][:, :, x0:x1].astype(np.int16)
    near = binary_dilation(lab[:, :, x0:x1] > 0, iterations=3)
    hu = np.where(near, hu, np.int16(-2048)).astype(np.int16)
    tmp = out / "seg15" / f"{cid}.tmp.npz"
    np.savez_compressed(tmp, lab=lab, off=lo, affine=aff15, labels=json.dumps(enc15.labels), hu_slab=hu,
                        slab_x0=np.int32(x0), shape15=np.array(lab15.shape))
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
    mask_dirs = [Path(d) for d in a.mask_dirs]
    lab3_names = json.loads((M3 / "dataset.json").read_text())["labels"]

    class E3:  # spine_roi only needs .labels
        labels = lab3_names

    enc15 = NnUNetEncoder(M15)
    logf = out / f"timing_w{a.wid}.csv"
    new = not logf.exists()
    fh = open(logf, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["id", "pull", "load", "roi_resample15", "seg15", "embed15", "save", "total_proc", "shape15",
                     "n15", "status", "peak_mb"])

    def next_ids():
        for cid in ids:
            if (out / "emb15" / f"{cid}.npz").exists():
                continue
            if claim(out, cid):
                yield cid

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

    n = 0
    cid, th = start_next()
    while cid is not None and (not a.max_cases or n < a.max_cases) and (not a.limit or n < a.limit):
        th.join()
        ncid, nth = (start_next() if (not a.max_cases or n + 1 < a.max_cases) and (not a.limit or n + 1 < a.limit)
                     else (None, None))
        T = {"pull": pre.get(cid + "_t", np.nan)}
        ct = pre.get(cid)
        torch.cuda.reset_peak_memory_stats()
        try:
            if isinstance(ct, Exception) or ct is None:
                raise RuntimeError(f"pull failed: {ct!r}")
            mp = next((d / f"{cid}.nii.gz" for d in mask_dirs if (d / f"{cid}.nii.gz").exists()), None)
            if mp is None:
                raise RuntimeError("no 3 mm mask")
            t0 = t = time.time()
            raw = nib.load(str(ct))
            img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header)
            img.header.set_slope_inter(1, 0)
            canon = as_closest_canonical(img)
            m3 = nib.load(str(mp))
            lab3 = np.asanyarray(m3.dataobj).astype(np.int32).transpose(2, 1, 0)
            T["load"] = time.time() - t; t = time.time()
            roi = P.spine_roi(canon, m3, lab3, E3)
            if roi is None:
                raise RuntimeError("no vertebrae in 3 mm mask")
            img15 = change_spacing(roi, list(enc15.spacing), order=1, dtype=np.int32, nr_cpus=a.threads)
            hu15 = np.asanyarray(img15.dataobj).astype(np.int16).transpose(2, 1, 0)
            del img, canon, raw
            T["roi_resample15"] = time.time() - t; t = time.time()
            lab15 = with_oom_retry(lambda: seg_nnunet(enc15, hu15))
            T["seg15"] = time.time() - t; t = time.time()
            units = P.units_from(lab15, enc15)
            r15 = P.embed_units(enc15, hu15, units, len(P.LEVELS), P.MIN_VOX["r15"])
            T["embed15"] = time.time() - t; t = time.time()
            if r15["present"].any():
                save_seg(out, cid, lab15, hu15, img15.affine, enc15, units)
            tmp = out / "emb15" / f"{cid}.tmp.npz"
            np.savez(tmp, levels=np.array(P.LEVELS), windows=np.array([w for w, _ in P.WINDOWS]),
                     stages=np.array(P.STAGES), shape15=np.array(hu15.shape), **r15)
            tmp.rename(out / "emb15" / f"{cid}.npz")
            T["save"] = time.time() - t
            T["total_proc"] = time.time() - t0
            status, s15, n15 = "ok", "x".join(map(str, hu15.shape)), int(r15["present"].sum())
        except Exception as e:  # noqa: BLE001
            status, s15, n15 = f"FAIL {e!r}"[:200].replace("\n", " "), "", 0
            with open(out / "failed.txt", "a") as ff:
                ff.write(f"{cid}\t{status}\n")
        finally:
            if not a.ct_local and isinstance(ct, Path):
                ct.unlink(missing_ok=True)
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 2**20
        wr.writerow([cid] + [f"{T.get(k, np.nan):.2f}" for k in ["pull", "load", "roi_resample15", "seg15",
                     "embed15", "save", "total_proc"]] + [s15, n15, status, f"{peak:.0f}"])
        fh.flush()
        print(f"w{a.wid} [{n+1}] {cid} {status} n15={n15} peak={peak:.0f}MB " +
              " ".join(f"{k}={v:.1f}" for k, v in T.items()), flush=True)
        n += 1
        cid, th = ncid, nth
    if cid is not None and th is not None:  # prefetched but not processed (life ended): release the claim
        th.join()
        ct = pre.get(cid)
        if not a.ct_local and isinstance(ct, Path):
            ct.unlink(missing_ok=True)
        (out / "claims" / cid).unlink(missing_ok=True)
    fh.close()
    if cid is None and n == 0:
        (out / f"DONE_w{a.wid}").touch()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prio")
    p.add_argument("--split", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--ids", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--mask-dirs", nargs="+", required=True)
    e.add_argument("--ct-local", default=None, help="read CTs from this dir instead of ssh-pulling from CT-rate")
    e.add_argument("--wid", type=int, default=0)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--max-cases", type=int, default=0)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--windows", default="all")
    a = ap.parse_args()
    {"prio": cmd_prio, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    main()
