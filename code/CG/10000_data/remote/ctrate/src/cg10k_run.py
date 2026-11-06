"""CG10k OWL extraction worker (E3_late expB inputs). One process = one worker; encoder loaded once.

Per case it runs the ORIGINAL pipeline functions unchanged:
  nib.load -> patches.resample_ct (TS-identical canonical + 3 mm) -> worker.seg_nnunet (TS --fast Dataset297)
  -> masks/{id}.nii.gz -> patches.finalize (prep/{id}.npz + patches/{id}.json, 12 main units)
  -> extract.embed_case (5 windows -> emb/<window_key>/{id}.npz)
  -> extract_bones.load_units + embed_bones (17 bone units x 3 windows -> emb_bones/<window_key>/{id}.npz)

Sets: --set NAME=IN_DIR=OUT_DIR (repeatable, processed in priority order). Case id = file stem (EXAMID).
Safety for the live rsync: dot/tmp files ignored; a file is used only after its size has been unchanged for
--stable-sec; unreadable/truncated gzip -> released and retried later (failed only after 6 tries).
Resume / no duplicates: done = all 5 emb + 3 bone npz exist; claims in STATE/<set>/claims/{id} (O_EXCL, holds
host:pid; a claim whose pid is dead is taken over). Exit: STATE/STOP or STATE/STOP_<tag>, or (with --watch) when
every set has no work left and is complete (IN_DIR/.SYNC_DONE exists, or no new file for --idle-exit-h hours).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import socket
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from common import ROOT, load_cfg, read_plans
from encoder import NnUNetEncoder
from extract import embed_case, window_key
from extract_bones import WINDOWS as BONE_WINDOWS, embed_bones, load_units
from patches import finalize, label_lookup, resample_ct, units_from_labels
from worker import seg_nnunet, with_oom_retry

HOST = socket.gethostname()
PRIMED = -1e18


class ReadError(Exception):
    pass


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class CaseSet:
    def __init__(self, spec, state_root, wins, bwins):
        self.name, inp, out = spec.split("=")
        self.inp, self.out = Path(inp), Path(out)
        for sub in ("masks", "prep", "patches", "emb", "emb_bones"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        for wk, _ in wins:
            (self.out / "emb" / wk).mkdir(parents=True, exist_ok=True)
        for wk, _ in bwins:
            (self.out / "emb_bones" / wk).mkdir(parents=True, exist_ok=True)
        self.state = state_root / self.name
        for sub in ("claims", "retry"):
            (self.state / sub).mkdir(parents=True, exist_ok=True)
        self.wins, self.bwins = wins, bwins
        self.seen = {}           # name -> (size, t_first_seen_with_this_size)
        self.last_new = None     # time a new file name was last seen
        self.retry_until = {}    # id -> time before which we do not retry
        self.done_cache = set()  # 11-06: ids already verified done (scan stat'ed 8 npz per done case: ~3.5 s/scan)

    def done(self, cid):
        return all((self.out / "emb" / wk / f"{cid}.npz").exists() for wk, _ in self.wins) and \
            all((self.out / "emb_bones" / wk / f"{cid}.npz").exists() for wk, _ in self.bwins)

    def failed_ids(self):
        f = self.state / "failed.jsonl"
        return {json.loads(l)["id"] for l in open(f)} if f.exists() else set()

    def scan(self, stable_sec):
        """-> list of ready, not-done, not-failed ids (sorted)."""
        now = time.time()
        ready = []
        try:
            names = os.listdir(self.inp)
        except FileNotFoundError:
            return []
        failed = self.failed_ids()
        for n in names:
            if n.startswith(".") or not n.endswith(".nii.gz") or ".tmp" in n:
                continue
            if n[:-7] in self.done_cache and n in self.seen:   # 11-06: finished cases never become undone
                continue
            try:
                sz = os.stat(self.inp / n).st_size
            except FileNotFoundError:
                continue
            prev = self.seen.get(n)
            if prev is None:
                self.last_new = now
            if prev is None or prev[0] != sz:
                self.seen[n] = (sz, now)
                continue
            if (prev[1] != PRIMED and now - prev[1] < stable_sec) or sz == 0:
                continue
            cid = n[:-7]
            if cid in failed or self.retry_until.get(cid, 0) > now:
                continue
            if self.done(cid):
                self.done_cache.add(cid)
                continue
            ready.append(cid)
        return sorted(ready)

    def prime(self):
        """files present at startup count as stable if their size is unchanged at the next scan."""
        self.scan(1e9)
        self.seen = {n: (v[0], PRIMED) for n, v in self.seen.items()}

    def complete(self, idle_h):
        if (self.inp / ".SYNC_DONE").exists():
            return True
        return self.last_new is not None and time.time() - self.last_new > idle_h * 3600

    def claim(self, cid):
        c = self.state / "claims" / cid
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                h, p = c.read_text().strip().split(":")
                if h == HOST and not pid_alive(int(p)):   # stale claim of a dead worker
                    c.unlink()
                    return self.claim(cid)
            except (ValueError, FileNotFoundError):
                pass
            return False
        os.write(fd, f"{HOST}:{os.getpid()}".encode())
        os.close(fd)
        if self.done(cid):        # finished by someone else between scan and claim
            c.unlink()
            return False
        return True

    def release(self, cid):
        try:
            (self.state / "claims" / cid).unlink()
        except FileNotFoundError:
            pass


GPU_LOCK = None  # 11-06: path of an flock file; GPU sections of all workers are serialized (no OOM contention)


class gpu_section:
    """exclusive use of the 3080 for one GPU step; empty_cache before release so the next holder has ~9 GB.
    Only scheduling changes - the pipeline functions run unchanged."""
    def __enter__(self):
        self.fd = None
        if GPU_LOCK is not None:
            import fcntl
            self.fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import fcntl
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        return False


def atomic_savez(dst: Path, arrs):
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, dst)


def process(cfg, cfg_b, enc, plans, lookup, S: CaseSet, cid, dev):
    cfg["out"] = cfg_b["out"] = S.out
    T = {}
    t = time.time()
    try:
        raw = nib.load(str(S.inp / f"{cid}.nii.gz"))
        img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header)  # decompress once (worker.py)
        img.header.set_slope_inter(1, 0)
    except Exception as e:  # noqa: BLE001  truncated / still-syncing file
        raise ReadError(repr(e)) from e
    T["read"] = time.time() - t
    t = time.time()
    img_r, hu = resample_ct(img, plans)
    T["resample"] = time.time() - t
    t = time.time()
    with gpu_section():
        lab = with_oom_retry(lambda: seg_nnunet(enc, hu))
    T["seg"] = time.time() - t
    t = time.time()
    mtmp = S.out / "masks" / f"{cid}.{os.getpid()}.tmp.nii.gz"
    nib.save(nib.Nifti1Image(lab.transpose(2, 1, 0), img_r.affine), str(mtmp))
    os.replace(mtmp, S.out / "masks" / f"{cid}.nii.gz")
    info = finalize(cfg, cid, hu, units_from_labels(cfg, lab, lookup), img_r.affine, False, plans)
    T["patches"] = time.time() - t
    del lab, img, raw
    t = time.time()
    ul = list(cfg["organ_map"].keys())
    with gpu_section():
        out = with_oom_retry(lambda: embed_case(cfg, enc, cid, S.wins, ul))
    for wk, arrs in out.items():
        atomic_savez(S.out / "emb" / wk / f"{cid}.npz", arrs)
    T["extract"] = time.time() - t
    t = time.time()
    hu2, units_b, regrid = load_units(cfg_b, cid, lookup)
    with gpu_section():
        res = with_oom_retry(lambda: embed_bones(cfg_b, enc, hu2, units_b, S.bwins, len(cfg_b["organ_map"]), 0.5))
    for wk, arrs in res.items():
        atomic_savez(S.out / "emb_bones" / wk / f"{cid}.npz", arrs)
    T["bones"] = time.time() - t
    if dev == "cuda":
        torch.cuda.empty_cache()
    a0 = next(iter(out.values()))
    b0 = next(iter(res.values()))
    return T, dict(shape="x".join(map(str, info["shape"])), n_main=int(a0["organ_present"].sum()),
                   n_bone=int(b0["organ_present"].sum()), tiles=int(a0["n_tiles"].sum()) + int(b0["n_tiles"].sum()),
                   regrid=int(regrid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_cg10k.yaml"))
    ap.add_argument("--set", action="append", required=True, help="NAME=IN_DIR=OUT_DIR, priority order")
    ap.add_argument("--state", default=str(ROOT / "state"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="w0", help="worker name (log file, STOP_<tag>)")
    ap.add_argument("--ids-file", default=None, help="restrict to these ids (smoke test)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--stable-sec", type=float, default=90)
    ap.add_argument("--idle-exit-h", type=float, default=6)
    ap.add_argument("--max-cases", type=int, default=0, help="exit with code 10 after this many cases (recycle)")
    ap.add_argument("--gpu-lock", default=None, help="flock file serializing the GPU steps of all workers")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global GPU_LOCK
    GPU_LOCK = a.gpu_lock if a.device == "cuda" else None
    cfg = load_cfg(a.config)
    cfg["device"] = a.device
    if a.device == "cpu":
        cfg["amp"] = False
    cfg_b = dict(cfg)
    cfg_b["organ_map"] = yaml.safe_load(open(ROOT / "organ_map_bones.yaml"))
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    wins = [(window_key(n, w), w) for n, w in cfg["windows"].items()]
    bwins = [(window_key(n, cfg["windows"][n]), cfg["windows"][n]) for n in BONE_WINDOWS]
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    sets = [CaseSet(s, state, wins, bwins) for s in a.set]
    for S in sets:
        S.prime()
    time.sleep(5)
    keep = {l.strip() for l in open(a.ids_file) if l.strip()} if a.ids_file else None
    enc = NnUNetEncoder(cfg["ts_model_dir"], device=a.device)
    plans, lookup = read_plans(cfg), label_lookup(cfg)
    logf = state / "timing.csv"
    cols = ["id", "set", "tag", "device", "threads", "read", "resample", "seg", "patches", "extract", "bones", "total",
            "maxrss_gib", "shape", "n_main", "n_bone", "tiles", "regrid", "end"]
    if not logf.exists():
        with open(logf, "w", newline="") as fh:
            csv.writer(fh).writerow(cols)
    print(f"{HOST} {a.tag} device={a.device} threads={a.threads} sets={[s.name for s in sets]} pid={os.getpid()}",
          flush=True)
    n = 0
    tries = {}
    consec = 0                # consecutive non-read failures -> circuit breaker (systematic error, e.g. CUDA)
    while True:
        if (state / "STOP").exists() or (state / f"STOP_{a.tag}").exists():
            print("STOP file -> exit", flush=True)
            break
        if a.max_cases and n >= a.max_cases:
            print(f"{a.tag} max-cases {n} reached -> exit 10 (recycle)", flush=True)
            sys.exit(10)
        got = None
        for S in sets:
            for cid in S.scan(a.stable_sec):
                if keep is not None and cid not in keep:
                    continue
                if S.claim(cid):
                    got = (S, cid)
                    break
            if got:
                break
        if got is None:
            if not a.watch or all(S.complete(a.idle_exit_h) for S in sets):
                # a final check: nothing claimable and (not watching or all sets complete)
                if not a.watch or not any(S.scan(0) for S in sets):
                    break
            time.sleep(20 if a.watch else 0)
            continue
        S, cid = got
        t0 = time.time()
        try:
            T, info = process(cfg, cfg_b, enc, plans, lookup, S, cid, a.device)
            T["total"] = time.time() - t0
            row = dict(id=cid, set=S.name, tag=a.tag, device=a.device, threads=a.threads, **{k: f"{v:.1f}" for k, v in T.items()},
                       maxrss_gib=f"{rss_gib():.2f}", end=time.strftime("%m-%d %H:%M:%S"), **info)
            with open(logf, "a", newline="") as fh:
                csv.DictWriter(fh, cols).writerow(row)
            n += 1
            consec = 0
            print(f"{a.tag} {S.name} {cid} " + " ".join(f"{k} {v:.1f}" for k, v in T.items()) +
                  f" rss {rss_gib():.1f}G main {info['n_main']}/12 bone {info['n_bone']}/17", flush=True)
        except Exception as e:  # noqa: BLE001
            k = (S.name, cid)
            tries[k] = tries.get(k, 0) + 1
            readerr = isinstance(e, ReadError)
            consec = 0 if readerr else consec + 1
            print(f"{a.tag} {S.name} {cid} ERROR try {tries[k]} {e!r}", flush=True)
            if (readerr and tries[k] < 6) or (not readerr and tries[k] < 2):
                S.retry_until[cid] = time.time() + (600 if readerr else 60)
            else:
                with open(S.state / "failed.jsonl", "a") as f:
                    f.write(json.dumps({"id": cid, "tag": a.tag, "err": repr(e),
                                        "tb": traceback.format_exc()[-1500:]}) + "\n")
            if a.device == "cuda":
                torch.cuda.empty_cache()
        finally:
            S.release(cid)
        if consec >= 5:
            print(f"{a.tag} 5 consecutive failures -> exit (check failed.jsonl; remove its entries to retry)", flush=True)
            sys.exit(3)
    print(f"{a.tag} exit after {n} cases", flush=True)


if __name__ == "__main__":
    main()
