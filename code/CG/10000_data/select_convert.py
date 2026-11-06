#!/usr/bin/env python
"""CG10k abdomen CT: pick one series per exam and convert it to NIfTI (HU, int16, 1.5x1.5x3 mm).

Input : /Volumes/PolypXfer/deliver_10k/images/2022/<PID>/<EXAMID>/*.dcm (mixed series per folder)
        report_meta_qwen.csv  -> only rows with abdomen == 1
Output: <out>/<EXAMID>.nii.gz           (RAS affine, int16 HU clipped to [-1024, 3071])
        <out>/records/<EXAMID>.json     (per-exam selection record; presence = done -> resumable)
        <out>/series_selection.csv      (rebuilt from records at the end / with --merge-only)

Series selection (per exam):
  eligible = ImageType ORIGINAL\\PRIMARY\\AXIAL, axial orientation, >= 20 slices, not
             scout/localizer/MPR/MIP/coronal/sagittal/lung/brain/CTA/bone (desc or kernel)
  phase    = delay | arterial | portal | cplus (unlabeled contrast) | noncontrast
             delay and arterial are never selected.
  rank     = 1 portal z>=300mm   2 cplus (unlabelled C+, e.g. 'Body C+') z>=300mm
             3 portal/cplus 200<=z<300 (flag limited_z)   4 noncontrast z>=300   5 noncontrast 200-300
  ties     = not chest-only desc, slice thickness in [2,5] mm, larger z-extent, thinner, later acquisition.
  flags    = limited_z, chest_series, multi_cplus_ambiguous, delay_full_available (tier>=3 and a
             full-coverage delay series exists), thick_slices, irregular_gaps, dup_positions, gantry_tilt_or_shift
  nothing  -> QC flag, record written, no NIfTI.

Spacing: the OWL pipeline (patches.resample_ct / TotalSegmentator --fast) does its own
as_closest_canonical + change_spacing to 3 mm iso, so we store 1.5x1.5x3 mm (the Merlin
preprocessing grid): z is resampled once (from the true slice positions, so irregular gaps
are handled), xy only to 1.5 mm (exact factor 2 to the 3 mm grid) to keep disk use ~5-8 MB/case.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/PolypXfer/deliver_10k/images/2022")
META = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_meta_qwen.csv")
OUT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/nifti_3mm")
XY, ZS = 1.5, 3.0
Z_FULL, Z_MIN = 300.0, 200.0
EXIT_DRIVE_GONE = 3  # exit code when the USB drive (ROOT) disappears mid-run


def drive_ok():
    """True while the USB drive is mounted and ROOT is listable (guards against a 2026-11-05-style disconnect
    turning every remaining exam into a bogus missing_dir/error/no_eligible record)."""
    try:
        return os.path.ismount("/Volumes/PolypXfer") and ROOT.is_dir()
    except OSError:
        return False


ABORT = {"_abort": "drive_gone"}  # returned by workers instead of writing a record

TAGS = ["SeriesNumber", "SeriesDescription", "SliceThickness", "ImageType", "SeriesInstanceUID",
        "ImagePositionPatient", "ImageOrientationPatient", "ConvolutionKernel", "ContrastBolusAgent",
        "Rows", "Columns", "PixelSpacing", "AcquisitionTime", "SeriesTime", "Modality"]

EXCL = re.compile(r"scout|locali|topo|surview|mpr|mip|minip|\bvr\b|\b3d\b|\bcor\b|coronal|\bsag\b|sagittal|lung|hrct|"
                  r"brain|head|h&n|\bhn\b|neck|cta|angio|aorta|bone|dose|report|screen|ivp|ctu|uro|kub|calcium|"
                  r"cardiac|heart|coronary|spine|orbit|sinus|temporal|\biac\b|pulmonary|\bpe\b|perfusion", re.I)
LUNG_KERNEL = re.compile(r"^(LUNG|BONE|BONEPLUS|DETAIL|EDGE|SHARP|FC5[0-9]|FC8[0-9]|B[6-9]\d|BR6\d|BL\d\d|I[6-9]\d)", re.I)
ART_T = {"a", "a1", "a2", "art", "arterial", "1st", "early", "aph", "late-art"}
POR_T = {"v", "p", "pv", "pvp", "portal", "venous", "2nd", "2st", "vph"}
DEL_T = {"d", "delay", "delayed", "dly", "late", "3rd", "equilibrium", "excretory", "hepatobiliary"}
ABD_RE = re.compile(r"abd|body|liver|pelv|kidney|renal|pancrea|gist|hcc|whole|c\.?a\.?p|chest\s*[+&-]\s*abd", re.I)


def fnum(x, d=np.nan):
    try:
        return float(x)
    except Exception:
        return d


def series_summary(ds_list):
    d0 = ds_list[0]
    desc = str(d0.get("SeriesDescription") or "").strip()
    itype = [str(x).upper() for x in (d0.get("ImageType") or [])]
    iop = d0.get("ImageOrientationPatient")
    n_unique, zext, normal_z, gaps = 0, 0.0, 0.0, []
    if iop is not None and len(iop) == 6:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        n = np.cross(r, c)
        normal_z = abs(n[2])
        pos = sorted({round(float(np.dot(np.array(x.ImagePositionPatient, float), n)), 3)
                      for x in ds_list if "ImagePositionPatient" in x})
        n_unique = len(pos)
        if n_unique > 1:
            zext = pos[-1] - pos[0]
            gaps = np.diff(pos)
    ps = d0.get("PixelSpacing")
    return {
        "uid": str(d0.get("SeriesInstanceUID")), "num": d0.get("SeriesNumber"), "desc": desc,
        "itype": "\\".join(itype), "n": len(ds_list), "n_unique": n_unique, "zext": round(zext, 1),
        "thick": fnum(d0.get("SliceThickness")), "kernel": str(d0.get("ConvolutionKernel") or ""),
        "agent": str(d0.get("ContrastBolusAgent") or ""), "normal_z": round(normal_z, 3),
        "ps": [float(x) for x in ps] if ps else None, "rows": d0.get("Rows"), "cols": d0.get("Columns"),
        "acq": str(d0.get("AcquisitionTime") or d0.get("SeriesTime") or ""),
        "gap_med": float(np.median(gaps)) if len(gaps) else np.nan,
        "gap_max": float(np.max(gaps)) if len(gaps) else np.nan,
        "modality": str(d0.get("Modality") or ""),
    }


def tokens(desc):
    d = re.sub(r"^\s*ce\s+", "", desc.lower())
    toks = [t.strip(".") for t in re.split(r"[\s\-_/,()]+", d) if t.strip(".")]
    out = set(toks)
    for t in toks:  # glued forms like 'c+1st', 'c+2st', '3p'
        for k in ("1st", "2nd", "2st", "3rd", "delay"):
            if k in t:
                out.add(k)
    return out


def classify(s):
    """Return (phase, reason_if_excluded)."""
    it = s["itype"].split("\\")
    if s["modality"] and s["modality"] != "CT":
        return None, "not_ct"
    if not ("ORIGINAL" in it and "PRIMARY" in it and "AXIAL" in it):
        return None, "imagetype"
    if s["normal_z"] < 0.9:
        return None, "not_axial"
    desc = s["desc"]
    if EXCL.search(desc):
        return None, "desc_excluded"
    k = s["kernel"].split("\\")[0].strip()
    if k and LUNG_KERNEL.match(k):
        return None, "lung_bone_kernel"
    if s["n_unique"] < 20:
        return None, "too_few_slices"
    if not s["ps"] or s["rows"] is None:
        return None, "no_geometry"
    tk = tokens(desc)
    if re.search(r"c\s*-|non[\s-]?con|pre[\s-]?con|w/o|without|plain", desc, re.I) and not re.search(r"c\s*\+", desc):
        return "noncontrast", None
    if tk & DEL_T:
        return "delay", None
    if tk & ART_T:
        return "arterial", None
    if tk & POR_T:
        return "portal", None
    if re.search(r"c\s*\+|\+\s*c\b|post|enhanc|contrast", desc, re.I) or s["agent"] not in ("", "None"):
        return "cplus", None
    return "unknown", None


def pick(series):
    """tiers: 1 portal z>=300 | 2 cplus z>=300 | 3 portal/cplus 200-300 | 4 noncontrast/unknown z>=300
              | 5 noncontrast/unknown 200-300.  Delay/arterial never chosen."""
    cands, flags = [], []
    for s in series:
        ph, why = classify(s)
        s["phase"], s["excl"] = ph, why
        if ph is None or ph in ("delay", "arterial"):
            continue
        z = s["zext"]
        full = z >= Z_FULL
        if z < Z_MIN:
            continue
        if ph in ("portal", "cplus"):
            tier = (1 if ph == "portal" else 2) if full else 3
        else:
            tier = 4 if full else 5
        th = s["thick"] if np.isfinite(s["thick"]) else 5.0
        chest_only = bool(re.search(r"chest|thorax", s["desc"], re.I)) and not ABD_RE.search(s["desc"])
        cands.append((tier, int(chest_only), 0 if 2.0 <= th <= 5.0 else 1, -z, th, _neg_time(s["acq"]), s))
    if not cands:
        return None, flags
    cands.sort(key=lambda x: x[:6])
    best = cands[0]
    s = best[-1]
    if s["zext"] < Z_FULL:
        flags.append("limited_z")
    if best[1]:
        flags.append("chest_series")
    if s["phase"] == "unknown":
        flags.append("phase_unknown")
    if s["phase"] == "cplus" and sum(1 for c in cands if c[0] == best[0] and c[-1]["phase"] == "cplus") > 1:
        flags.append("multi_cplus_ambiguous")
    if best[0] >= 3 and any(x["phase"] == "delay" and x["zext"] >= Z_FULL for x in series):
        flags.append("delay_full_available")
    if np.isfinite(s["thick"]) and s["thick"] > 5.0:
        flags.append("thick_slices")
    if np.isfinite(s["gap_med"]) and s["gap_max"] > 1.6 * s["gap_med"]:
        flags.append("irregular_gaps")
    if s["n"] != s["n_unique"]:
        flags.append("dup_positions")
    s["tier"] = best[0]
    return s, flags


def _neg_time(t):
    try:
        return -float(t)
    except Exception:
        return 0.0


def read_headers(exam_dir):
    """One pass over the exam folder: each file is read whole (the drive is seek-bound, so reading the
    full ~200 KB costs about the same as reading the header) and kept in memory for the pixel pass."""
    import io
    import pydicom
    groups = {}
    for f in os.listdir(exam_dir):
        if f.startswith("."):
            continue
        p = os.path.join(exam_dir, f)
        try:
            with open(p, "rb") as fh:
                raw = fh.read()
            d = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, specific_tags=TAGS)
        except Exception:
            continue
        d.raw_ = raw
        groups.setdefault(str(d.get("SeriesInstanceUID")), []).append(d)
    return groups


def load_volume(ds_list):
    """Read pixels -> HU volume (k, row, col) sorted along slice normal + positions + geometry."""
    import pydicom
    d0 = ds_list[0]
    iop = np.array(d0.ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    items = []
    for d in ds_list:
        ipp = np.array(d.ImagePositionPatient, float)
        items.append((float(np.dot(ipp, n)), id(d), d))
    items.sort()
    # drop duplicate positions (keep first)
    uniq, last = [], None
    for pos, _, dd in items:
        if last is None or abs(pos - last) > 1e-3:
            uniq.append((pos, dd))
            last = pos
    slices, ipps, poss = [], [], []
    import io
    for pos, dd in uniq:
        d = pydicom.dcmread(io.BytesIO(dd.raw_))
        a = d.pixel_array.astype(np.float32)
        a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
        slices.append(a)
        ipps.append(np.array(d.ImagePositionPatient, float))
        poss.append(pos)
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise RuntimeError(f"inconsistent slice shapes {shapes}")
    vol = np.stack(slices, 0)
    ps = [float(x) for x in d0.PixelSpacing]  # [row spacing, col spacing]
    return vol, np.array(poss), np.array(ipps), r, c, n, ps


def resample(vol, poss, ps):
    """z: linear interp at 3 mm from true positions; xy: linear to 1.5 mm (no upsampling)."""
    from scipy.ndimage import affine_transform
    t = np.arange(poss[0], poss[-1] + 1e-6, ZS)
    idx = np.clip(np.searchsorted(poss, t, side="right") - 1, 0, len(poss) - 2)
    w = ((t - poss[idx]) / (poss[idx + 1] - poss[idx])).astype(np.float32)
    w = np.clip(w, 0, 1)[:, None, None]
    vz = vol[idx] * (1 - w) + vol[idx + 1] * w
    sy, sx = max(ps[0], XY), max(ps[1], XY)
    if sy == ps[0] and sx == ps[1]:
        return vz, t, ps
    R, C = vol.shape[1:]
    nr, nc = int(round((R - 1) * ps[0] / sy)) + 1, int(round((C - 1) * ps[1] / sx)) + 1
    out = affine_transform(vz, np.array([1.0, sy / ps[0], sx / ps[1]]), output_shape=(len(t), nr, nc),
                           order=1, mode="nearest", prefilter=False)
    return out, t, [sy, sx]


def save_nifti(vol, t, ipp0, r, c, n, ps_new, path):
    import nibabel as nib
    # voxel (i=col, j=row, k=slice) -> LPS
    lps = np.eye(4)
    lps[:3, 0] = r * ps_new[1]
    lps[:3, 1] = c * ps_new[0]
    lps[:3, 2] = n * ZS
    lps[:3, 3] = ipp0 + (t[0] - float(np.dot(ipp0, n))) * n
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    data = np.clip(np.rint(vol), -1024, 3071).astype(np.int16).transpose(2, 1, 0)
    img = nib.Nifti1Image(data, aff)
    img.set_qform(aff, code=1)
    img.set_sform(aff, code=1)
    img.header.set_xyzt_units("mm")
    tmp = path.with_name(path.name + ".tmp.nii.gz")
    nib.save(img, str(tmp))
    os.replace(tmp, path)
    return data.shape


def process(job):
    pid, exid, out = job
    out = Path(out)
    rec_p = out / "records" / f"{exid}.json"
    if rec_p.exists():
        return json.loads(rec_p.read_text())
    if not drive_ok():
        return dict(ABORT, EXAMID=exid)
    t0 = time.time()
    rec = {"EXAMID": exid, "PID": pid, "status": "", "flags": [], "SeriesDescription": "", "SeriesNumber": "",
           "SeriesInstanceUID": "", "phase": "", "tier": "", "n_slices": "", "z_extent_mm": "",
           "slice_thickness": "", "src_pixel_spacing": "", "out_shape": "", "out_spacing": "",
           "hu_min": "", "hu_max": "", "kernel": "", "n_series": 0, "n_files": 0, "sec": 0}
    try:
        ed = ROOT / pid / exid
        if not ed.is_dir():
            rec["status"], rec["flags"] = "skip", ["missing_dir"]
        else:
            groups = read_headers(str(ed))
            series = [series_summary(v) for v in groups.values()]
            rec["n_series"], rec["n_files"] = len(series), sum(s["n"] for s in series)
            s, flags = pick(series)
            rec["series"] = [{k: x.get(k) for k in ("num", "desc", "itype", "n", "zext", "thick", "kernel", "phase", "excl")}
                             for x in sorted(series, key=lambda x: fnum(x["num"], 0))]
            rec["flags"] = flags
            if s is None:
                phases = {x["phase"] for x in series}
                rec["status"] = "skip"
                rec["flags"].append("no_eligible_series" + ("_only_delay_or_arterial" if phases & {"delay", "arterial"} else ""))
            else:
                sel = groups[s["uid"]]
                for u in list(groups):
                    if u != s["uid"]:
                        del groups[u]  # free raw bytes of the other series
                vol, poss, ipps, r, c, n, ps = load_volume(sel)
                tilt = np.linalg.norm(np.cross(ipps[-1] - ipps[0], n)) if len(ipps) > 1 else 0
                if tilt > 2.0:
                    rec["flags"].append("gantry_tilt_or_shift")
                v, t, ps_new = resample(vol, poss, ps)
                shape = save_nifti(v, t, ipps[0], r, c, n, ps_new, out / f"{exid}.nii.gz")
                rec.update(status="ok", SeriesDescription=s["desc"], SeriesNumber=s["num"], SeriesInstanceUID=s["uid"],
                           phase=s["phase"], tier=s["tier"], n_slices=s["n_unique"], z_extent_mm=s["zext"],
                           slice_thickness=s["thick"], src_pixel_spacing=f"{ps[0]:.4f}x{ps[1]:.4f}x{np.median(np.diff(poss)):.3f}",
                           out_shape="x".join(map(str, shape)), out_spacing=f"{ps_new[1]:.3f}x{ps_new[0]:.3f}x{ZS}",
                           hu_min=int(np.floor(vol.min())), hu_max=int(np.ceil(vol.max())), kernel=s["kernel"])
    except Exception as e:
        rec["status"] = "error"
        rec["flags"].append("error:" + type(e).__name__ + ":" + str(e)[:200])
        rec["traceback"] = traceback.format_exc()[-1500:]
    if rec["status"] != "ok" and not drive_ok():
        return dict(ABORT, EXAMID=exid)  # drive vanished while processing: write nothing, redo next run
    rec["sec"] = round(time.time() - t0, 1)
    tmp = rec_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, rec_p)
    return rec


COLS = ["EXAMID", "PID", "status", "SeriesDescription", "SeriesNumber", "SeriesInstanceUID", "phase", "tier",
        "n_slices", "z_extent_mm", "slice_thickness", "src_pixel_spacing", "out_shape", "out_spacing",
        "hu_min", "hu_max", "kernel", "n_series", "n_files", "flags", "sec"]


def merge(out):
    recs = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    with open(out / "series_selection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            r = dict(r)
            r["flags"] = ";".join(r.get("flags") or [])
            w.writerow(r)
    return recs


def summarize(recs):
    from collections import Counter
    st = Counter(r["status"] for r in recs)
    ph = Counter(r["phase"] for r in recs if r["status"] == "ok")
    tier = Counter(r["tier"] for r in recs if r["status"] == "ok")
    fl = Counter(f.split(":")[0] for r in recs for f in r.get("flags") or [])
    print(f"records={len(recs)} status={dict(st)} phase={dict(ph)} tier={dict(tier)} flags={dict(fl)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated EXAMIDs")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        summarize(merge(out))
        return
    if not drive_ok():
        print(f"{ROOT} not available -> abort", flush=True)
        sys.exit(EXIT_DRIVE_GONE)
    import pandas as pd
    m = pd.read_csv(META, usecols=["PID", "EXAMID", "abdomen"])
    m = m[m.abdomen == 1].drop_duplicates("EXAMID")
    if a.ids:
        m = m[m.EXAMID.isin(a.ids.split(","))]
    jobs = [(p, e, str(out)) for p, e in zip(m.PID, m.EXAMID) if not (out / "records" / f"{e}.json").exists()]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(m)} abdomen exams, {len(jobs)} to do, workers={a.workers}", flush=True)
    from multiprocessing import Pool
    t0, k = time.time(), 0
    with Pool(a.workers) as pool:
        for rec in pool.imap_unordered(process, jobs):
            if rec.get("_abort") or not drive_ok():
                pool.terminate()
                print(f"{time.strftime('%F %T')} DRIVE GONE ({ROOT} missing) at {rec['EXAMID']} after {k} cases "
                      f"-> abort, no record written; rerun to resume", flush=True)
                merge(out)
                os._exit(EXIT_DRIVE_GONE)
            k += 1
            el = time.time() - t0
            print(f"[{k}/{len(jobs)}] {rec['EXAMID']} {rec['status']} {rec.get('phase','')} "
                  f"'{rec.get('SeriesDescription','')}' z={rec.get('z_extent_mm','')} {rec.get('out_shape','')} "
                  f"{';'.join(rec.get('flags') or [])} {rec['sec']}s | {el/k:.2f}s/case eta {(len(jobs)-k)*el/k/3600:.2f}h",
                  flush=True)
            if k % 200 == 0:
                merge(out)
    summarize(merge(out))


if __name__ == "__main__":
    main()
