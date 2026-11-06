#!/usr/bin/env python
"""CG10k: header-only DICOM file index for ALL 10,000 exams (every .dcm, every series).

Input : /Volumes/PolypXfer/deliver_10k/images/2022/<PID>/<EXAMID>/*  (hash-named .dcm, mixed series)
        report_meta_qwen.csv (PID, EXAMID; all 10,000 rows, not only abdomen)
Output: <out>/shards/<EXAMID>.parquet   one row per file (presence = exam done -> resumable)
        <out>/files.parquet             all shards merged
        <out>/series.csv                one row per (EXAMID, SeriesInstanceUID)
        <out>/exams.csv                 one row per exam (n_files, n_series, n_err, sec)

Per-file columns: PID, EXAMID, rel_path (relative to ROOT: <PID>/<EXAMID>/<file>), file_size,
  StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID, SeriesNumber, SeriesDescription, ImageType,
  InstanceNumber, ipp_x, ipp_y, ipp_z (ImagePositionPatient), ImageOrientationPatient, SliceThickness,
  PixelSpacing, Rows, Columns, ConvolutionKernel, ContrastBolusAgent, AcquisitionTime, SeriesTime,
  Modality, BodyPartExamined, StudyDescription, error ('' = ok).
  Multi-valued tags are stored as backslash-joined strings.  An exam whose folder is missing gets one
  row with rel_path='' and error='missing_dir'.

Only headers are read (pydicom stop_before_pixels + specific_tags, force=True); the drive is a USB
spinning disk, so NEVER run this while another job reads /Volumes/PolypXfer (see wait_then_index.sh).

Lookup:  from build_dicom_index import series_files
         series_files("EX0015862", "1.2.392....496204")  -> [abs paths sorted by z / InstanceNumber]
CLI:     python build_dicom_index.py --lookup EX0015862 1.2.392....496204
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/Volumes/PolypXfer/deliver_10k/images/2022")
META = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_meta_qwen.csv")
OUT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/dicom_index")
NIFTI_RECORDS = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/nifti_3mm/records")

TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesNumber", "SeriesDescription",
        "ImageType", "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "SliceThickness",
        "PixelSpacing", "Rows", "Columns", "ConvolutionKernel", "ContrastBolusAgent", "AcquisitionTime",
        "SeriesTime", "Modality", "BodyPartExamined", "StudyDescription"]
STR_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription", "ImageType",
            "ImageOrientationPatient", "PixelSpacing", "ConvolutionKernel", "ContrastBolusAgent",
            "AcquisitionTime", "SeriesTime", "Modality", "BodyPartExamined", "StudyDescription"]
NUM_TAGS = ["SeriesNumber", "InstanceNumber", "SliceThickness", "Rows", "Columns"]
COLS = (["PID", "EXAMID", "rel_path", "file_size"] + STR_TAGS[:3] + ["SeriesNumber", "SeriesDescription",
        "ImageType", "InstanceNumber", "ipp_x", "ipp_y", "ipp_z", "ImageOrientationPatient", "SliceThickness",
        "PixelSpacing", "Rows", "Columns", "ConvolutionKernel", "ContrastBolusAgent", "AcquisitionTime",
        "SeriesTime", "Modality", "BodyPartExamined", "StudyDescription", "error"])


def _s(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "\\".join(str(x) for x in v)
    return str(v).strip()


def _f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def read_one(path, rel, size):
    import pydicom
    row = {c: None for c in COLS}
    row.update(rel_path=rel, file_size=size, error="")
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS, force=True)
        for t in STR_TAGS:
            row[t] = _s(d.get(t))
        for t in NUM_TAGS:
            row[t] = _f(d.get(t))
        ipp = d.get("ImagePositionPatient")
        if ipp is not None and len(ipp) == 3:
            row["ipp_x"], row["ipp_y"], row["ipp_z"] = (_f(x) for x in ipp)
        if not row["SeriesInstanceUID"]:
            row["error"] = "no_SeriesInstanceUID"
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def _frame(rows):
    import pandas as pd
    df = pd.DataFrame(rows, columns=COLS)
    for c in ["file_size"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in NUM_TAGS + ["ipp_x", "ipp_y", "ipp_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in COLS:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str)
    return df


def index_exam(job):
    """Read every header of one exam; write shards/<EXAMID>.parquet atomically. Returns a small summary."""
    pid, exid, out, root = job
    shard = Path(out) / "shards" / f"{exid}.parquet"
    t0 = time.time()
    rows = []
    try:
        ed = Path(root) / pid / exid
        if not ed.is_dir():
            rows.append({"rel_path": "", "file_size": None, "error": "missing_dir"})
        else:
            with os.scandir(ed) as it:
                ents = sorted((e for e in it if not e.name.startswith(".") and e.is_file()), key=lambda e: e.name)
            for e in ents:
                rows.append(read_one(e.path, f"{pid}/{exid}/{e.name}", e.stat().st_size))
            if not ents:
                rows.append({"rel_path": "", "file_size": None, "error": "empty_dir"})
    except Exception as e:
        # exam-level failure (e.g. I/O error listing the folder): no shard -> retried on the next run
        return {"EXAMID": exid, "PID": pid, "n_files": 0, "n_err": 1, "n_series": 0, "bytes": 0,
                "sec": round(time.time() - t0, 2), "first_error": f"exam_error(no shard) {type(e).__name__}: {str(e)[:150]}"}
    if not Path(root).is_dir():  # drive unmounted mid-run: do not mark the exam done with bogus errors
        raise RuntimeError(f"root {root} disappeared while indexing {exid}; aborting (rerun resumes)")
    for r in rows:
        r["PID"], r["EXAMID"] = pid, exid
    df = _frame(rows)
    tmp = shard.with_name(shard.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, shard)
    ok = df[df.error == ""]
    return {"EXAMID": exid, "PID": pid, "n_files": int((df.rel_path != "").sum()), "n_err": int((df.error != "").sum()),
            "n_series": int(ok.SeriesInstanceUID.nunique()), "bytes": int(df.file_size.fillna(0).sum()),
            "sec": round(time.time() - t0, 2), "first_error": (df.error[df.error != ""].iloc[0] if len(df) > len(ok) else "")}


# ---------------------------------------------------------------- series summary / merge
def _mode(s):
    s = s[s != ""] if s.dtype == object else s.dropna()
    if not len(s):
        return "" if s.dtype == object else float("nan")
    return s.value_counts().index[0]


def series_table(df):
    import numpy as np
    import pandas as pd
    ok = df[(df.error == "") & (df.SeriesInstanceUID != "")]
    out = []
    for (exid, uid), g in ok.groupby(["EXAMID", "SeriesInstanceUID"], sort=False):
        z = g.ipp_z.dropna()
        zu = np.unique(np.round(z.values, 3))
        gaps = np.diff(zu)
        out.append({
            "PID": g.PID.iloc[0], "EXAMID": exid, "SeriesInstanceUID": uid,
            "StudyInstanceUID": _mode(g.StudyInstanceUID), "SeriesNumber": _mode(g.SeriesNumber),
            "SeriesDescription": _mode(g.SeriesDescription), "ImageType": _mode(g.ImageType),
            "n_imagetypes": g.ImageType.nunique(), "Modality": _mode(g.Modality),
            "n_files": len(g), "n_unique_z": len(zu),
            "z_min": float(zu[0]) if len(zu) else np.nan, "z_max": float(zu[-1]) if len(zu) else np.nan,
            "z_extent": float(zu[-1] - zu[0]) if len(zu) else np.nan,
            "z_gap_med": float(np.median(gaps)) if len(gaps) else np.nan,
            "z_gap_max": float(np.max(gaps)) if len(gaps) else np.nan,
            "SliceThickness": _mode(g.SliceThickness), "n_thickness": g.SliceThickness.nunique(),
            "PixelSpacing": _mode(g.PixelSpacing), "Rows": _mode(g.Rows), "Columns": _mode(g.Columns),
            "ImageOrientationPatient": _mode(g.ImageOrientationPatient),
            "ConvolutionKernel": _mode(g.ConvolutionKernel), "ContrastBolusAgent": _mode(g.ContrastBolusAgent),
            "SeriesTime": _mode(g.SeriesTime), "AcquisitionTime_min": g.AcquisitionTime[g.AcquisitionTime != ""].min()
            if (g.AcquisitionTime != "").any() else "",
            "BodyPartExamined": _mode(g.BodyPartExamined), "StudyDescription": _mode(g.StudyDescription),
            "instance_min": g.InstanceNumber.min(), "instance_max": g.InstanceNumber.max(),
            "bytes": int(g.file_size.sum()),
        })
    s = pd.DataFrame(out)
    if len(s):
        s = s.sort_values(["EXAMID", "SeriesNumber", "SeriesInstanceUID"], kind="stable")
    return s


def merge(out):
    import pandas as pd
    shards = sorted((out / "shards").glob("*.parquet"))
    if not shards:
        print("no shards", flush=True)
        return None
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    tmp = out / "files.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out / "files.parquet")
    s = series_table(df)
    s.to_csv(out / "series.csv.tmp", index=False)
    os.replace(out / "series.csv.tmp", out / "series.csv")
    g = df.groupby("EXAMID")
    ex = pd.DataFrame({"PID": g.PID.first(), "n_files": g.rel_path.apply(lambda x: (x != "").sum()),
                       "n_err": g.error.apply(lambda x: (x != "").sum()),
                       "n_series": g.SeriesInstanceUID.apply(lambda x: x[x != ""].nunique()),
                       "bytes": g.file_size.sum(), "errors": g.error.apply(lambda x: ";".join(sorted(set(x[x != ""])))[:300])})
    ex.reset_index().to_csv(out / "exams.csv", index=False)
    print(f"merged {len(shards)} exams: {len(df)} file rows, {(df.error != '').sum()} error rows, "
          f"{len(s)} series -> {out}/files.parquet, series.csv, exams.csv", flush=True)
    return df, s


# ---------------------------------------------------------------- lookup
def series_files(examid, series_uid, out=OUT, root=ROOT, sort=True):
    """Absolute paths of all files of one series, sorted by ipp_z then InstanceNumber.
    Uses the per-exam shard (fast); falls back to files.parquet with a filter."""
    import pandas as pd
    out = Path(out)
    shard = out / "shards" / f"{examid}.parquet"
    if shard.exists():
        df = pd.read_parquet(shard)
    else:
        df = pd.read_parquet(out / "files.parquet", filters=[("EXAMID", "==", examid)])
    df = df[(df.SeriesInstanceUID == series_uid) & (df.error == "")]
    if sort:
        df = df.sort_values(["ipp_z", "InstanceNumber", "rel_path"], na_position="last")
    return [str(Path(root) / r) for r in df.rel_path]


# ---------------------------------------------------------------- run
def smoke_check(summ, out):
    """Validate a freshly indexed exam against the nifti_3mm selection record when one exists."""
    import json
    import pandas as pd
    exid = summ["EXAMID"]
    df = pd.read_parquet(out / "shards" / f"{exid}.parquet")
    s = series_table(df)
    msg = [f"SMOKE {exid}: files={summ['n_files']} err={summ['n_err']} series={summ['n_series']} {summ['sec']}s"]
    for _, r in s.iterrows():
        msg.append(f"   #{r.SeriesNumber:g} n={r.n_files} z=[{r.z_min:.1f},{r.z_max:.1f}] th={r.SliceThickness} "
                   f"'{r.SeriesDescription}' {r.ImageType}")
    rp = NIFTI_RECORDS / f"{exid}.json"
    if rp.exists():
        rec = json.loads(rp.read_text())
        chk = [f"n_files idx={summ['n_files'] - summ['n_err']} rec={rec.get('n_files')}",
               f"n_series idx={summ['n_series']} rec={rec.get('n_series')}"]
        uid = rec.get("SeriesInstanceUID")
        if uid:
            sel = s[s.SeriesInstanceUID == uid]
            paths = series_files(exid, uid, out=out)
            chk.append(f"selected series n_unique_z idx={int(sel.n_unique_z.iloc[0]) if len(sel) else None} "
                       f"rec={rec.get('n_slices')} lookup_paths={len(paths)} first_exists={os.path.exists(paths[0]) if paths else None}")
        msg.append("   vs nifti_3mm record: " + " | ".join(chk))
    else:
        msg.append("   (no nifti_3mm record for this exam; non-abdomen or not converted)")
    print("\n".join(msg), flush=True)


def run_jobs(jobs, workers, out, tag=""):
    from multiprocessing import Pool
    t0, k, nf = time.time(), 0, 0
    if not jobs:
        return 0.0, 0.0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(index_exam, jobs):
            k += 1
            nf += r["n_files"]
            el = time.time() - t0
            print(f"{tag}[{k}/{len(jobs)}] {r['EXAMID']} files={r['n_files']} series={r['n_series']} err={r['n_err']} "
                  f"{r['first_error'][:80]} {r['sec']}s | {el / k:.2f}s/case {nf / el:.1f} files/s "
                  f"eta {(len(jobs) - k) * el / k / 3600:.2f}h", flush=True)
            if not tag and k % 2500 == 0:
                merge(out)
    el = time.time() - t0
    return el / k, nf / el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated EXAMIDs")
    ap.add_argument("--smoke", type=int, default=0, help="first N exams sequentially with detailed checks")
    ap.add_argument("--bench", type=int, default=0, help="N exams per worker count in --bench-workers; pick fastest")
    ap.add_argument("--bench-workers", default="1,2,3")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--lookup", nargs=2, metavar=("EXAMID", "SERIES_UID"))
    a = ap.parse_args()
    out = Path(a.out)
    if a.lookup:
        print("\n".join(series_files(*a.lookup, out=out, root=a.root)))
        return
    (out / "shards").mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        merge(out)
        return
    import pandas as pd
    if not Path(a.root).is_dir():
        sys.exit(f"root not found: {a.root}")
    m = pd.read_csv(a.meta, usecols=["PID", "EXAMID"], dtype=str).drop_duplicates("EXAMID")
    if a.ids:
        m = m[m.EXAMID.isin(a.ids.split(","))]
    todo = [(p, e, str(out), a.root) for p, e in zip(m.PID, m.EXAMID)
            if not (out / "shards" / f"{e}.parquet").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{time.strftime('%F %T')} {len(m)} exams in meta, {len(todo)} to do", flush=True)

    if a.smoke and todo:
        # prefer exams that already have a nifti_3mm record so the index can be cross-checked
        has = [j for j in todo if (NIFTI_RECORDS / f"{j[1]}.json").exists()][: a.smoke]
        smoke = has + [j for j in todo if j not in has][: a.smoke - len(has)]
        todo = smoke + [j for j in todo if j not in smoke]
        print(f"--- smoke test on {len(smoke)} exams (sequential)", flush=True)
        for j in smoke:
            try:
                smoke_check(index_exam(j), out)
            except Exception:
                print("SMOKE FAILED:\n" + traceback.format_exc(), flush=True)
                sys.exit(2)
        todo = todo[len(smoke):]
        merge(out)

    workers = a.workers
    if a.bench and todo:
        res = {}
        for w in [int(x) for x in a.bench_workers.split(",")]:
            chunk, todo = todo[: a.bench], todo[a.bench:]
            if not chunk:
                break
            spc, fps = run_jobs(chunk, w, out, tag=f"bench w={w} ")
            res[w] = fps
            print(f"--- bench workers={w}: {spc:.2f}s/case {fps:.1f} files/s", flush=True)
        if res:
            best = max(res, key=res.get)
            # stay with the default unless another count is clearly (>10%) faster
            workers = best if res[best] > 1.10 * res.get(a.workers, 0) else a.workers
            print(f"--- bench result {res} -> workers={workers}", flush=True)

    print(f"{time.strftime('%F %T')} main pass: {len(todo)} exams, workers={workers}", flush=True)
    run_jobs(todo, workers, out)
    merge(out)
    print(f"{time.strftime('%F %T')} done", flush=True)


if __name__ == "__main__":
    main()
