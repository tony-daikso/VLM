#!/usr/bin/env python
"""Non-contrast (C-) pass for the CG10k C+ vs C- paired comparison (2026-11-06).

For every exam in owl_infer/probs_main.csv whose main series is contrast-enhanced (phase portal/cplus in
nifti_3mm/series_selection.csv), pick the best eligible non-contrast abdominal axial series from the DICOM header
index (dicom_index/series.csv, built by build_dicom_index.py) and convert it exactly like select_convert.py
(1.5x1.5x3 mm, int16 HU, RAS) -> nifti_3mm_nc/<EXAMID>.nii.gz, records/<EXAMID>.json, series_selection_nc.csv.
Only the files of the chosen series are read (paths from the index shard) -> no folder scan of the USB drive.

Eligibility = select_convert.classify(...) == "noncontrast" (ORIGINAL/PRIMARY/AXIAL, axial, >= 20 slices, no
lung/bone kernel, scout/MPR/CTA/... excluded by description), z-extent >= 200 mm.
Rank: z >= 300 mm first (else flag limited_z), not chest-only description, slice thickness in [2,5] mm, larger z,
thinner, later acquisition (= select_convert.pick tie-breaks).
Resumable (record present = done). Drive guard: if /Volumes/PolypXfer disappears -> no record, exit 3.
Exams whose main series already is C- (152) are not converted: their C- prediction is the main one.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import select_convert as sc
import build_dicom_index as bdi

BASE = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
OUT = BASE / "nifti_3mm_nc"
IDX = BASE / "dicom_index"


def _str(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def summary_from_index(r):
    """dicom_index/series.csv row -> the dict select_convert.classify expects."""
    iop = [float(x) for x in _str(r["ImageOrientationPatient"]).split("\\") if x.strip()] \
        if _str(r["ImageOrientationPatient"]) else []
    nz = 0.0
    if len(iop) == 6:
        nz = abs(float(np.cross(np.array(iop[:3]), np.array(iop[3:]))[2]))
    ps = _str(r["PixelSpacing"])
    ps = [float(x) for x in ps.split("\\")] if ps else None
    return {
        "uid": _str(r["SeriesInstanceUID"]), "num": r["SeriesNumber"], "desc": _str(r["SeriesDescription"]).strip(),
        "itype": _str(r["ImageType"]).upper(), "n": int(sc.fnum(r["n_files"])), "n_unique": int(sc.fnum(r["n_unique_z"])),
        "zext": round(sc.fnum(r["z_extent"], 0.0), 1), "thick": sc.fnum(r["SliceThickness"]),
        "kernel": _str(r["ConvolutionKernel"]), "agent": _str(r["ContrastBolusAgent"]), "normal_z": round(nz, 3),
        "ps": ps, "rows": None if np.isnan(sc.fnum(r["Rows"])) else int(sc.fnum(r["Rows"])),
        "cols": None if np.isnan(sc.fnum(r["Columns"])) else int(sc.fnum(r["Columns"])),
        "acq": _str(r["AcquisitionTime_min"]) or _str(r["SeriesTime"]),
        "gap_med": sc.fnum(r["z_gap_med"]), "gap_max": sc.fnum(r["z_gap_max"]), "modality": _str(r["Modality"]),
    }


def pick_nc(series):
    cands = []
    for s in series:
        ph, why = sc.classify(s)
        s["phase"], s["excl"] = ph, why
        if ph != "noncontrast" or s["zext"] < sc.Z_MIN:
            continue
        th = s["thick"] if np.isfinite(s["thick"]) else 5.0
        chest_only = bool(re.search(r"chest|thorax", s["desc"], re.I)) and not sc.ABD_RE.search(s["desc"])
        cands.append((0 if s["zext"] >= sc.Z_FULL else 1, int(chest_only), 0 if 2.0 <= th <= 5.0 else 1,
                      -s["zext"], th, sc._neg_time(s["acq"]), s))
    if not cands:
        return None, []
    cands.sort(key=lambda x: x[:6])
    best = cands[0]
    s = best[-1]
    flags = []
    if best[0]:
        flags.append("limited_z")
    if best[1]:
        flags.append("chest_series")
    if sum(1 for c in cands if c[0] == best[0]) > 1:
        flags.append("multi_nc")
    if np.isfinite(s["thick"]) and s["thick"] > 5.0:
        flags.append("thick_slices")
    if np.isfinite(s["gap_med"]) and np.isfinite(s["gap_max"]) and s["gap_max"] > 1.6 * s["gap_med"]:
        flags.append("irregular_gaps")
    if s["n"] != s["n_unique"]:
        flags.append("dup_positions")
    s["tier"] = 4 if best[0] == 0 else 5
    return s, flags


def plan(sel_csv, probs_csv, series_csv):
    """-> list of (PID, EXAMID, chosen-series dict, flags, main_uid) for every main C+ exam with an eligible C- series,
    plus a counter of reasons for the others."""
    import pandas as pd
    main = set(pd.read_csv(probs_csv, usecols=["study_id"]).study_id)
    sel = pd.read_csv(sel_csv)
    sel = sel[(sel.status == "ok") & sel.EXAMID.isin(main)]
    cplus = sel[sel.phase.isin(["portal", "cplus"])]
    ser = pd.read_csv(series_csv, dtype={"SeriesInstanceUID": str, "SeriesDescription": str, "ImageType": str,
                                         "ImageOrientationPatient": str, "PixelSpacing": str,
                                         "ConvolutionKernel": str, "ContrastBolusAgent": str, "Modality": str,
                                         "AcquisitionTime_min": str, "SeriesTime": str})
    ser = ser[ser.EXAMID.isin(set(cplus.EXAMID))]
    by = {e: g for e, g in ser.groupby("EXAMID")}
    jobs, why = [], {}
    for r in cplus.itertuples():
        g = by.get(r.EXAMID)
        if g is None:
            why["not_in_index"] = why.get("not_in_index", 0) + 1
            continue
        s, flags = pick_nc([summary_from_index(x) for _, x in g.iterrows()])
        if s is None:
            why["no_eligible_nc"] = why.get("no_eligible_nc", 0) + 1
            continue
        if s["uid"] == r.SeriesInstanceUID:
            why["same_as_main"] = why.get("same_as_main", 0) + 1
            continue
        jobs.append((r.PID, r.EXAMID, s, flags, r.SeriesInstanceUID))
    return jobs, why, len(cplus)


def read_series(examid, uid):
    import pydicom
    paths = bdi.series_files(examid, uid, out=IDX, root=sc.ROOT)
    out = []
    for p in paths:
        with open(p, "rb") as fh:
            raw = fh.read()
        d = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, specific_tags=sc.TAGS)
        d.raw_ = raw
        out.append(d)
    return out, len(paths)


def process(job):
    pid, exid, s, flags, main_uid, out = job
    out = Path(out)
    rec_p = out / "records" / f"{exid}.json"
    if rec_p.exists():
        return json.loads(rec_p.read_text())
    if not sc.drive_ok():
        return dict(sc.ABORT, EXAMID=exid)
    t0 = time.time()
    rec = {k: "" for k in sc.COLS}
    rec.update(EXAMID=exid, PID=pid, flags=list(flags), n_series="", n_files=0, main_SeriesInstanceUID=main_uid)
    try:
        dss, nf = read_series(exid, s["uid"])
        rec["n_files"] = nf
        if not dss:
            raise RuntimeError("no files in index for series")
        vol, poss, ipps, r, c, n, ps = sc.load_volume(dss)
        if len(ipps) > 1 and np.linalg.norm(np.cross(ipps[-1] - ipps[0], n)) > 2.0:
            rec["flags"].append("gantry_tilt_or_shift")
        v, t, ps_new = sc.resample(vol, poss, ps)
        shape = sc.save_nifti(v, t, ipps[0], r, c, n, ps_new, out / f"{exid}.nii.gz")
        rec.update(status="ok", SeriesDescription=s["desc"], SeriesNumber=s["num"], SeriesInstanceUID=s["uid"],
                   phase="noncontrast", tier=s["tier"], n_slices=s["n_unique"], z_extent_mm=s["zext"],
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


COLS = sc.COLS + ["main_SeriesInstanceUID"]


def merge(out):
    recs = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    with open(out / "series_selection_nc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            r = dict(r)
            r["flags"] = ";".join(r.get("flags") or [])
            w.writerow(r)
    return recs


def smoke_compare(a, n):
    import nibabel as nib
    import pandas as pd
    if not sc.drive_ok():
        print("drive not available", flush=True)
        return sc.EXIT_DRIVE_GONE
    sel = pd.read_csv(a.sel)
    sel = sel[(sel.status == "ok") & (sel.phase == "noncontrast") & sel.flags.isna()].head(n)
    ser = pd.read_csv(a.series, dtype=str)
    sm = Path(a.out) / "_smoke"
    (sm / "records").mkdir(parents=True, exist_ok=True)
    bad = 0
    for r in sel.itertuples():
        g = ser[ser.EXAMID == r.EXAMID]
        g = g.astype({"n_files": int, "n_unique_z": int})
        s, flags = pick_nc([summary_from_index(x) for _, x in g.iterrows()])
        same = s is not None and s["uid"] == r.SeriesInstanceUID
        (sm / "records" / f"{r.EXAMID}.json").unlink(missing_ok=True)
        rec = process((r.PID, r.EXAMID, s, flags, r.SeriesInstanceUID, str(sm))) if s else {"status": "no_pick"}
        ok = False
        if rec.get("status") == "ok":
            x, y = nib.load(sm / f"{r.EXAMID}.nii.gz"), nib.load(OUT.parent / "nifti_3mm" / f"{r.EXAMID}.nii.gz")
            ok = x.shape == y.shape and np.allclose(x.affine, y.affine, atol=1e-3) and \
                np.array_equal(np.asarray(x.dataobj), np.asarray(y.dataobj))
        print(f"smoke {r.EXAMID} same_series={same} status={rec.get('status')} identical_to_main={ok} "
              f"sec={rec.get('sec')}", flush=True)
        bad += not (same and ok)
    print(f"smoke: {len(sel) - bad}/{len(sel)} identical", flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sel", default=str(BASE / "nifti_3mm/series_selection.csv"))
    ap.add_argument("--probs", default=str(BASE / "owl_infer/probs_main.csv"))
    ap.add_argument("--series", default=str(IDX / "series.csv"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="print the selection counts, read nothing from the drive")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--smoke-compare", type=int, default=0,
                    help="convert N exams whose MAIN series is C- via the index path into <out>/_smoke and compare "
                         "voxel-wise with nifti_3mm/<id>.nii.gz; exit 1 on mismatch")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        merge(out)
        return
    if a.smoke_compare:
        sys.exit(smoke_compare(a, a.smoke_compare))
    jobs, why, n_cplus = plan(a.sel, a.probs, a.series)
    from collections import Counter
    fl = Counter(f for j in jobs for f in j[3])
    print(f"{time.strftime('%F %T')} main C+ exams {n_cplus}: with eligible C- {len(jobs)} "
          f"(z>=300 {sum(1 for j in jobs if j[2]['tier'] == 4)}, 200-300 {sum(1 for j in jobs if j[2]['tier'] == 5)}); "
          f"others {why}; flags {dict(fl)}", flush=True)
    if a.plan_only:
        import pandas as pd
        pd.DataFrame([dict(EXAMID=j[1], desc=j[2]["desc"], z=j[2]["zext"], thick=j[2]["thick"], kernel=j[2]["kernel"],
                           flags=";".join(j[3])) for j in jobs]).to_csv(out / "plan_nc.csv", index=False)
        return
    if not sc.drive_ok():
        print(f"{sc.ROOT} not available -> abort", flush=True)
        sys.exit(sc.EXIT_DRIVE_GONE)
    todo = [(p, e, s, f, mu, str(out)) for p, e, s, f, mu in jobs if not (out / "records" / f"{e}.json").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} to do, workers={a.workers}", flush=True)
    from multiprocessing import Pool
    t0, k = time.time(), 0
    with Pool(a.workers) as pool:
        for rec in pool.imap_unordered(process, todo):
            if rec.get("_abort") or not sc.drive_ok():
                pool.terminate()
                print(f"{time.strftime('%F %T')} DRIVE GONE ({sc.ROOT} missing) at {rec['EXAMID']} after {k} cases "
                      f"-> abort, no record written; rerun to resume", flush=True)
                merge(out)
                os._exit(sc.EXIT_DRIVE_GONE)
            k += 1
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {rec['EXAMID']} {rec['status']} '{rec.get('SeriesDescription', '')}' "
                  f"z={rec.get('z_extent_mm', '')} {rec.get('out_shape', '')} {';'.join(rec.get('flags') or [])} "
                  f"{rec['sec']}s | {el / k:.2f}s/case eta {(len(todo) - k) * el / k / 3600:.2f}h", flush=True)
            if k % 200 == 0:
                merge(out)
    recs = merge(out)
    print(f"{time.strftime('%F %T')} wall {time.time() - t0:.0f}s records={len(recs)} "
          f"status={dict(Counter(r['status'] for r in recs))} "
          f"flags={dict(Counter(f.split(':')[0] for r in recs for f in r.get('flags') or []))}", flush=True)
    if any(r["status"] == "error" for r in recs):
        print("some errors (see records/*.json traceback)", flush=True)


if __name__ == "__main__":
    main()
