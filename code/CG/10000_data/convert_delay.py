#!/usr/bin/env python
"""Second pass: for exams flagged delay_full_available in series_selection.csv, convert the full-coverage
delay-phase series too (same 1.5x1.5x3 mm, int16 HU, RAS) -> nifti_3mm_delay/<EXAMID>.nii.gz,
records/<EXAMID>.json, series_selection_delay.csv. Resumable. Run only after select_convert.py finished
(both read the same USB drive).

Delay pick: phase == delay (select_convert.classify), z-extent >= 300 mm; ties: slice thickness in [2,5] mm,
larger z-extent, thinner, later acquisition.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import select_convert as sc

BASE = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
OUT = BASE / "nifti_3mm_delay"
SEL = BASE / "nifti_3mm" / "series_selection.csv"


def pick_delay(series):
    cands = []
    for s in series:
        ph, why = sc.classify(s)
        s["phase"], s["excl"] = ph, why
        if ph != "delay" or s["zext"] < sc.Z_FULL:
            continue
        th = s["thick"] if np.isfinite(s["thick"]) else 5.0
        cands.append((0 if 2.0 <= th <= 5.0 else 1, -s["zext"], th, sc._neg_time(s["acq"]), s))
    if not cands:
        return None, []
    cands.sort(key=lambda x: x[:4])
    s = cands[0][-1]
    flags = []
    if len(cands) > 1:
        flags.append("multi_delay")
    if np.isfinite(s["thick"]) and s["thick"] > 5.0:
        flags.append("thick_slices")
    if np.isfinite(s["gap_med"]) and s["gap_max"] > 1.6 * s["gap_med"]:
        flags.append("irregular_gaps")
    if s["n"] != s["n_unique"]:
        flags.append("dup_positions")
    return s, flags


def process(job):
    pid, exid, out = job
    out = Path(out)
    rec_p = out / "records" / f"{exid}.json"
    if rec_p.exists():
        return json.loads(rec_p.read_text())
    if not sc.drive_ok():
        return dict(sc.ABORT, EXAMID=exid)
    t0 = time.time()
    rec = {k: "" for k in sc.COLS}
    rec.update(EXAMID=exid, PID=pid, flags=[], n_series=0, n_files=0)
    try:
        groups = sc.read_headers(str(sc.ROOT / pid / exid))
        series = [sc.series_summary(v) for v in groups.values()]
        rec["n_series"], rec["n_files"] = len(series), sum(s["n"] for s in series)
        s, flags = pick_delay(series)
        rec["flags"] = flags
        if s is None:
            rec["status"] = "skip"
            rec["flags"].append("no_full_delay_series")
        else:
            sel = groups[s["uid"]]
            vol, poss, ipps, r, c, n, ps = sc.load_volume(sel)
            if len(ipps) > 1 and np.linalg.norm(np.cross(ipps[-1] - ipps[0], n)) > 2.0:
                rec["flags"].append("gantry_tilt_or_shift")
            v, t, ps_new = sc.resample(vol, poss, ps)
            shape = sc.save_nifti(v, t, ipps[0], r, c, n, ps_new, out / f"{exid}.nii.gz")
            rec.update(status="ok", SeriesDescription=s["desc"], SeriesNumber=s["num"], SeriesInstanceUID=s["uid"],
                       phase="delay", tier="delay", n_slices=s["n_unique"], z_extent_mm=s["zext"],
                       slice_thickness=s["thick"],
                       src_pixel_spacing=f"{ps[0]:.4f}x{ps[1]:.4f}x{np.median(np.diff(poss)):.3f}",
                       out_shape="x".join(map(str, shape)), out_spacing=f"{ps_new[1]:.3f}x{ps_new[0]:.3f}x{sc.ZS}",
                       hu_min=int(np.floor(vol.min())), hu_max=int(np.ceil(vol.max())), kernel=s["kernel"])
    except Exception as e:
        rec["status"] = "error"
        rec["flags"].append("error:" + type(e).__name__ + ":" + str(e)[:200])
        rec["traceback"] = traceback.format_exc()[-1500:]
    if rec["status"] != "ok" and not sc.drive_ok():
        return dict(sc.ABORT, EXAMID=exid)  # drive vanished while processing: write nothing, redo next run
    rec["sec"] = round(time.time() - t0, 1)
    tmp = rec_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, rec_p)
    return rec


def merge(out):
    recs = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    with open(out / "series_selection_delay.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, sc.COLS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            r = dict(r)
            r["flags"] = ";".join(r.get("flags") or [])
            w.writerow(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sel", default=str(SEL))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    if not a.merge_only:
        if not sc.drive_ok():
            print(f"{sc.ROOT} not available -> abort", flush=True)
            sys.exit(sc.EXIT_DRIVE_GONE)
        rows = list(csv.DictReader(open(a.sel)))
        rows = [r for r in rows if "delay_full_available" in (r["flags"] or "").split(";")]
        jobs = [(r["PID"], r["EXAMID"], str(out)) for r in rows
                if not (out / "records" / f"{r['EXAMID']}.json").exists()]
        print(f"{len(rows)} delay_full_available exams, {len(jobs)} to do, workers={a.workers}", flush=True)
        from multiprocessing import Pool
        t0, k = time.time(), 0
        with Pool(a.workers) as pool:
            for rec in pool.imap_unordered(process, jobs):
                if rec.get("_abort") or not sc.drive_ok():
                    pool.terminate()
                    print(f"{time.strftime('%F %T')} DRIVE GONE ({sc.ROOT} missing) at {rec['EXAMID']} after {k} "
                          f"cases -> abort, no record written; rerun to resume", flush=True)
                    merge(out)
                    os._exit(sc.EXIT_DRIVE_GONE)
                k += 1
                el = time.time() - t0
                print(f"[{k}/{len(jobs)}] {rec['EXAMID']} {rec['status']} '{rec.get('SeriesDescription','')}' "
                      f"z={rec.get('z_extent_mm','')} {rec.get('out_shape','')} {';'.join(rec.get('flags') or [])} "
                      f"{rec['sec']}s | {el/k:.2f}s/case eta {(len(jobs)-k)*el/k/3600:.2f}h", flush=True)
        print(f"wall {time.time()-t0:.0f}s", flush=True)
    recs = merge(out)
    from collections import Counter
    print(f"records={len(recs)} status={dict(Counter(r['status'] for r in recs))} "
          f"flags={dict(Counter(f.split(':')[0] for r in recs for f in r.get('flags') or []))}", flush=True)


if __name__ == "__main__":
    main()
