"""Host-side worker for the streamed full run: masks + prep + patches with ONE read of each CT.

Inbox protocol (filled by feeder.py on the Mac):
  inbox/{id}.nii.gz   CT copied to this host (deleted after processing)
  inbox/{id}.ref      text file holding a local CT path (host already has the data; never deleted)
  inbox/STOP          no more work will come; workers exit once the inbox is empty
A worker claims an item by atomic rename into inbox/claimed/w{wid}/. Outputs: {out}/masks, prep, patches.
Done marker = patches/{id}.json. Failures: {out}/failed.jsonl and inbox/failed/.
Run N copies with different --wid.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from common import load_cfg, read_plans
from patches import finalize, label_lookup, process, resample_ct, units_from_labels
from windowing import apply_window


def claim(inbox: Path, wid: int):
    for p in sorted(inbox.iterdir()):
        if p.name.startswith(".") or not (p.name.endswith(".nii.gz") or p.name.endswith(".ref")):
            continue
        dst = inbox / "claimed" / f"w{wid}" / p.name
        try:
            os.rename(p, dst)
            return dst
        except FileNotFoundError:
            continue  # another worker took it
    return None


def with_oom_retry(fn, tries=3):
    for attempt in range(tries):  # CUDA OOM on small GPUs shared by several workers: wait and retry
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


@torch.no_grad()
def seg_nnunet(enc, hu):
    """TotalSegmentator --fast prediction without the TS wrapper: baseline CTNormalization (verified identical to
    nnU-Net's preprocessor) -> nnU-Net sliding window (same predictor settings as TS) -> argmax. (z,y,x) uint8."""
    x = torch.from_numpy(apply_window(hu, {"type": "baseline"}, enc.norm)).float()[None]
    try:
        logits = enc.predictor.predict_sliding_window_return_logits(x)
    except torch.OutOfMemoryError:  # very large volume: accumulate the 118-class logits in host RAM instead
        torch.cuda.empty_cache()
        enc.predictor.perform_everything_on_device = False
        try:
            logits = enc.predictor.predict_sliding_window_return_logits(x)
        finally:
            enc.predictor.perform_everything_on_device = True
    lab = logits.argmax(0).to(torch.uint8).cpu().numpy()
    del logits
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--out", default=None, help="override output_dir for this host")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=0,
                    help="exit after this many cases (run_host.sh restarts the worker) so the CUDA caching "
                         "allocator's fragmented peak memory is returned; 0 = never")
    ap.add_argument("--engine", default="nnunet", choices=["nnunet", "ts"],
                    help="nnunet: predictor loaded once, one resample, in-process sliding window (fast); "
                         "ts: TotalSegmentator python API per case (original, ~3-4x slower)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.out:
        cfg["out"] = Path(a.out)
        for sub in ("masks", "prep", "patches"):
            (cfg["out"] / sub).mkdir(parents=True, exist_ok=True)
    inbox = Path(a.inbox)
    (inbox / "claimed" / f"w{a.wid}").mkdir(parents=True, exist_ok=True)
    (inbox / "failed").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if a.engine == "ts":
        from totalsegmentator.python_api import totalsegmentator
    else:
        from encoder import NnUNetEncoder
        enc = NnUNetEncoder(cfg["ts_model_dir"], device=cfg["device"])

    plans, lookup = read_plans(cfg), label_lookup(cfg)
    host = socket.gethostname()
    ts_device = "gpu" if cfg["device"].startswith("cuda") else cfg["device"]  # cuda->gpu; mps/cpu as is
    n = 0
    while True:
        if a.max_cases and n >= a.max_cases:
            break
        item = claim(inbox, a.wid)
        if item is None:
            if (inbox / "STOP").exists():
                break
            time.sleep(5)
            continue
        is_ref = item.name.endswith(".ref")
        cid = item.name.split(".")[0]
        t = time.time()
        try:
            src = Path(item.read_text().strip()) if is_ref else item
            raw = nib.load(str(src))
            img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header)  # decompress once
            img.header.set_slope_inter(1, 0)
            t_read = time.time() - t
            if a.engine == "nnunet":
                img_r, hu = resample_ct(img, plans)
                lab = with_oom_retry(lambda: seg_nnunet(enc, hu))
                nib.save(nib.Nifti1Image(lab.transpose(2, 1, 0), img_r.affine), str(cfg["out"] / "masks" / f"{cid}.nii.gz"))
                t_ts = time.time() - t - t_read
                finalize(cfg, cid, hu, units_from_labels(cfg, lab, lookup), img_r.affine, False, plans)
                item.unlink()
                n += 1
                torch.cuda.empty_cache()
                print(f"w{a.wid} {cid} read {t_read:.1f}s seg {t_ts:.1f}s total {time.time()-t:.1f}s", flush=True)
                continue
            for attempt in range(3):  # CUDA OOM on small GPUs shared by several workers: wait and retry
                try:
                    seg = totalsegmentator(img, None, ml=True, fast=bool(cfg.get("ts_fast")), task=cfg["ts_task"],
                                           save_lowres=bool(cfg.get("ts_fast")), device=ts_device, quiet=True,
                                           nr_thr_resamp=1, nr_thr_saving=1)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise
                    time.sleep(20 * (attempt + 1))
            seg = nib.Nifti1Image(np.asanyarray(seg.dataobj).astype(np.uint8), seg.affine)
            nib.save(seg, str(cfg["out"] / "masks" / f"{cid}.nii.gz"))
            t_ts = time.time() - t - t_read
            process(cfg, cid, plans, lookup, img=img, msk=seg)
            item.unlink()  # copied CT is deleted; for .ref only the ticket is removed
            n += 1
            torch.cuda.empty_cache()  # release the caching allocator's peak so other workers fit
            print(f"w{a.wid} {cid} read {t_read:.1f}s ts {t_ts:.1f}s total {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            with open(cfg["out"] / "failed.jsonl", "a") as f:
                f.write(json.dumps({"id": cid, "host": host, "wid": a.wid, "err": repr(e)}) + "\n")
            try:
                os.rename(item, inbox / "failed" / item.name)
            except OSError:  # item vanished (e.g. removed by hand) — never let error handling kill the worker
                pass
            torch.cuda.empty_cache()
            print(f"w{a.wid} {cid} FAILED {e!r}", flush=True)
    print(f"w{a.wid} exit after {n} cases", flush=True)


if __name__ == "__main__":
    main()
