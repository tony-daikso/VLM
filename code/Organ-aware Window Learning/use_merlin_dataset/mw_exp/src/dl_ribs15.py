"""Exp L Azure downloader (runs on the 4090): keeps a bounded buffer of raw Merlin CTs for
rib15_full.py extract --consume (which deletes each CT after processing).

The container SAS URL is read from a 600 file and handed to curl through a config on stdin (never on argv, never
logged). Ids are taken in --ids order, skipping ones that already have emb15/, are claimed (claims synced from the
A6000 by the Mac relay) or are in the buffer.
usage: python dl_ribs15.py --ids ids_4090.txt --out /root/OWL/out_ribs15 --buf /root/L/buf --par 32 --max-files 160
stop: touch <buf>/STOP_DL
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buf", required=True)
    ap.add_argument("--sas-file", default="/root/OWL/.azure_sas")
    ap.add_argument("--subdir", default="merlin_data")
    ap.add_argument("--par", type=int, default=32)
    ap.add_argument("--max-files", type=int, default=160)
    ap.add_argument("--max-gb", type=float, default=35.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = Path(a.sas_file).read_text().strip()
    base, q = s.split("?", 1)
    out, buf = Path(a.out), Path(a.buf)
    buf.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    log = open(buf.parent / "dl.log", "a")
    inflight, lock = set(), threading.Lock()
    stats = {"n": 0, "bytes": 0, "fail": 0, "t0": time.time()}

    def get(cid):
        t = time.time()
        part, dst = buf / f".{cid}.part", buf / f"{cid}.nii.gz"
        cfg = (f'url = "{base}/{a.subdir}/{cid}.nii.gz?{q}"\noutput = "{part}"\nsilent\nfail\nretry = 3\n'
               f'retry-delay = 5\nspeed-limit = 20000\nspeed-time = 120\nmax-time = 2400\n')
        r = subprocess.run(["curl", "-K", "-"], input=cfg, text=True, capture_output=True)
        try:
            if r.returncode == 0:
                part.rename(dst)
                sz = dst.stat().st_size
                with lock:
                    stats["n"] += 1; stats["bytes"] += sz
                print(f"{time.strftime('%T')} got {cid} {sz / 1e6:.0f}MB {time.time() - t:.1f}s", file=log, flush=True)
            else:
                part.unlink(missing_ok=True)
                with lock:
                    stats["fail"] += 1
                print(f"{time.strftime('%T')} FAIL {cid} rc={r.returncode}", file=log, flush=True)
        finally:
            with lock:
                inflight.discard(cid)

    ex = ThreadPoolExecutor(a.par)
    failed_once = set()
    submitted = 0
    last_rep = 0.0
    while not (buf / "STOP_DL").exists():
        have = {p.name[:-7] for p in buf.glob("*.nii.gz")}
        for cid in list(have):   # buffered CTs done / taken elsewhere (claim older than 10 min) -> free the slot
            c = out / "claims" / cid
            if (out / "emb15" / f"{cid}.npz").exists() or (c.exists() and time.time() - c.stat().st_mtime > 600):
                (buf / f"{cid}.nii.gz").unlink(missing_ok=True); have.discard(cid)
        bsize = sum(p.stat().st_size for p in buf.glob("*.nii.gz") if p.exists()) / 1e9
        done = {p.name[:-4] for p in (out / "emb15").glob("*.npz")} | {p.name for p in (out / "claims").iterdir()}
        with lock:
            room = min(a.max_files - len(have) - len(inflight), a.par - len(inflight))
            todo = [c for c in ids if c not in done and c not in have and c not in inflight and c not in failed_once]
        if bsize > a.max_gb:
            room = 0
        if not todo and not inflight:
            break
        for cid in todo[: max(0, room)]:
            if a.limit and submitted >= a.limit:
                break
            with lock:
                inflight.add(cid)
            submitted += 1
            ex.submit(get, cid)
        if a.limit and submitted >= a.limit and not inflight:
            break
        if time.time() - last_rep > 60:
            dt = time.time() - stats["t0"]
            print(f"{time.strftime('%T')} STAT got {stats['n']} fail {stats['fail']} {stats['bytes'] / 1e6 / dt:.1f} MB/s "
                  f"buf {len(have)} files {bsize:.1f} GB inflight {len(inflight)}", file=log, flush=True)
            last_rep = time.time()
        time.sleep(2)
    ex.shutdown(wait=True)
    if not (buf / "STOP_DL").exists() and not a.limit:
        (buf / "DONE_DOWNLOAD").touch()
    dt = time.time() - stats["t0"]
    print(f"{time.strftime('%T')} END got {stats['n']} fail {stats['fail']} {stats['bytes'] / 1e6 / dt:.1f} MB/s", file=log, flush=True)


if __name__ == "__main__":
    main()
