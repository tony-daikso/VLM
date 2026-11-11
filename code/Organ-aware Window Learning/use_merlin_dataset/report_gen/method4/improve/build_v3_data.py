#!/usr/bin/env python3
"""Agent G, T_unit_gt_v3 data: data_unit_v2 prompts + m4ext deterministic measurements (report_gen/m4ext, train+val only).

Changes to each prompt (everything else, incl. the completed classifier line, is kept character for character):
  1. unit line, kidney part: per-kidney hydronephrosis score = logistic stack of [logit(probe score), log1p(flcc_ml),
     flcc_ap_mm / 10, flcc-missing flag] (flcc = renal pelvis / collecting-system fluid proxy, patient side).
     TRAIN: 5-fold OOF with the region_units/hydronephrosis folds (the probe scores are already OOF with the same folds);
     VAL: stack refit on all train. Stack inputs and labels = improve/probe (unit_cue.json labels_train: report side).
     unit_scores.kid is replaced too, and unit_cue_v3.json gets the kidney t_f1 / pos-neg pools of the stacked TRAIN OOF
     scores, so train.py --unit-cue-gt rebuilds the gt line on the same scale. Vertebra part unchanged (probe).
  2. measurement line: "; <L1|L2> trabecular bone density <bmd_hu> HU" (m4ext bmd_hu: L1 ROI, L2 fallback, empty with
     cement -> "trabecular bone density unknown").
  3. new line after the measurement line, vertebral body geometry (1.5 mm): the T8-L5 levels whose mid-body height loss
     (g15_loss_m) is >= t_loss, largest first (at most 3), each "L1 height loss 32%, anterior/posterior height ratio 0.71";
     else "no T8-L5 level with height loss >= x%". t_loss = F1-optimal level threshold on TRAIN level labels only
     (unit_cue.json). The line is deterministic (same at train and inference) and in practice lists levels mostly in
     fracture-positive cases; it is not gated on the (train-time gt) fracture cue, which would differ between train and
     inference.
Numbers are only ever thresholded / fitted on train; val numbers are reported, never used.
Outputs (A6000, hold case ids, never in git): <out>/{train,val}.jsonl, <out>/build_v3_stats.json, <val600-out>,
<probe-out>/{unit_cue_v3.json, stack_kidney.csv}.
usage: python build_v3_data.py --m4ext m4ext_trval.csv --probe improve/probe --in improve/data_unit_v2 \
         --out improve/data_unit_v3 --ids data_sub/val_qwen_subset_ids.txt --val600-out improve/data/val600_unit_v3.jsonl
"""
import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import MEAS_PREFIX, UNIT_LEVELS, UNIT_PREFIX, render_unit_line, replace_unit_line  # noqa: E402

FOLD_CSV = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/hydronephrosis/out/oof_train.csv"
GEO_PREFIX = "Vertebral body geometry (1.5 mm): "
SIDES = (("right", "kidney_r"), ("left", "kidney_l"))   # probe order (right, left); m4ext kidney_l = patient left


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def auroc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    ss = s[o]
    i = 0
    while i < len(ss):                            # average ranks for ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def f1_threshold(y, s, grid=None):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    best = (0.0, 0.5, 0.0, 0.0)
    for t in (grid if grid is not None else np.unique(np.round(s, 2))):
        p = s >= t
        tp = float((p & (y == 1)).sum())
        f1 = 2 * tp / max(p.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(t), tp / max(p.sum(), 1), tp / max((y == 1).sum(), 1))
    return best[1], best[0], best[2], best[3]


def fit_logistic(X, y, lam=1e-2, iters=50):
    """L2 logistic regression by Newton / IRLS on standardised features -> (mu, sd, w, b)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1])
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def predict_logistic(m, X):
    mu, sd, w = m
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))


def val_kidney_labels():
    """(case, side index) -> 1/0 for val kidneys, unit_probe.py rule (report side; case owl-negative -> both 0)."""
    from m4_common import DISEASES, owl_binary
    hy = DISEASES.index("hydronephrosis")
    m4 = Path("/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4")
    out = {}
    for line in open(m4 / "unit_labels/unit_labels_val.jsonl"):
        r = json.loads(line)
        for s, (side, _) in enumerate(SIDES):
            for d, v in r["units"].get(f"organ3/kidney_{side}", []):
                if d == hy:
                    out[(r["case_id"], s)] = float(v)
    for line in open("/home/advantech/Desktop/Tony/VLM/OWL/data/labels_disease14_all.jsonl"):
        r = json.loads(line)
        if r.get("split") != "val":
            continue                                         # test rows skipped unread
        ans = r["answer"] if isinstance(r["answer"], dict) else json.loads(r["answer"])
        if owl_binary(ans["hydronephrosis"]["category"]) == 0:
            out[(str(r["study_id"]), 0)] = out[(str(r["study_id"]), 1)] = 0.0
    return out


def kid_features(p, ext, col):
    flcc, ap = fnum(ext.get(f"{col}_flcc_ml")), fnum(ext.get(f"{col}_flcc_ap_mm"))
    p = min(max(p, 1e-4), 1 - 1e-4)
    return [math.log(p / (1 - p)), math.log1p(flcc or 0.0), (ap or 0.0) / 10.0, 1.0 if flcc is None else 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4ext", required=True)
    ap.add_argument("--probe", required=True, help="improve/probe (probe_scores.csv, unit_cue.json)")
    ap.add_argument("--in", dest="inp", required=True, help="data_unit_v2 dir (train.jsonl, val.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-out", default="", help="default: <probe>_v3")
    ap.add_argument("--ids", required=True, help="val600 id list")
    ap.add_argument("--val600-out", required=True)
    ap.add_argument("--max-levels", type=int, default=3)
    a = ap.parse_args()
    probe = Path(a.probe)
    pout = Path(a.probe_out or str(probe) + "_v3")
    pout.mkdir(parents=True, exist_ok=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)

    need = {"study_id", "split", "bmd_hu", "bmd_level"}
    for _, c in SIDES:
        need |= {f"{c}_flcc_ml", f"{c}_flcc_ap_mm"}
    for lv in UNIT_LEVELS:
        need |= {f"{lv}_g15_loss_m", f"{lv}_g15_wedge"}
    ext = {}
    with open(a.m4ext) as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train", "val"):
                ext[r["study_id"]] = {k: r[k] for k in need if k in r}
    sc = {r["case_id"]: r for r in csv.DictReader(open(probe / "probe_scores.csv"))}
    uc = json.load(open(probe / "unit_cue.json"))
    lab = uc["labels_train"]
    fold = {r["study_id"]: int(r["fold"]) for r in csv.DictReader(open(FOLD_CSV))}
    st = {"n_ext": len(ext)}

    # ---------------------------------------------------------------- 1. kidney stack
    units = []                                     # (case, side index, split, fold, x, y)
    for cid, r in sc.items():
        sp = r["split"]
        e = ext.get(cid, {})
        for s, (side, col) in enumerate(SIDES):
            p = fnum(r["kid_" + side])
            if p is None:
                continue
            y = lab.get(cid, {}).get("kid", [None, None])[s] if sp == "train" else None
            fd = fold.get(cid, zlib.crc32(cid.encode()) % 5) if sp == "train" else -1
            units.append((cid, s, sp, fd, kid_features(p, e, col), y, p))
    X = np.array([u[4] for u in units])
    Y = np.array([np.nan if u[5] is None else float(u[5]) for u in units])
    SP = np.array([u[2] for u in units])
    FD = np.array([u[3] for u in units])
    P0 = np.array([u[6] for u in units])
    out = np.full(len(units), np.nan)
    tr = SP == "train"
    for k in range(5):
        fit = tr & (FD != k) & ~np.isnan(Y)
        m = fit_logistic(X[fit], Y[fit])
        out[tr & (FD == k)] = predict_logistic(m, X[tr & (FD == k)])
    mfull = fit_logistic(X[tr & ~np.isnan(Y)], Y[tr & ~np.isnan(Y)])
    va = SP == "val"
    out[va] = predict_logistic(mfull, X[va])
    st["kidney_stack"] = {"features": ["logit(probe)", "log1p(flcc_ml)", "flcc_ap_mm/10", "flcc_missing"],
                          "weights_std_full_train": [round(float(w), 3) for w in mfull[2]],
                          "train_units_labelled": int((tr & ~np.isnan(Y)).sum()), "train_pos": int(np.nansum(Y[tr])),
                          "train_oof_auroc_probe": auroc(Y[tr], P0[tr]), "train_oof_auroc_stack": auroc(Y[tr], out[tr])}
    tk, f1k, _, _ = f1_threshold(Y[tr], out[tr])
    st["kidney_stack"]["t_f1_train"], st["kidney_stack"]["f1_train"] = tk, f1k
    newkid = {}
    for u, v in zip(units, out):
        newkid.setdefault(u[0], [None, None])[u[1]] = round(float(v), 2)
    # side accuracy on train OOF (positive cases with both sides labelled), probe vs stack, same rule as unit_probe.py
    for nm, S, t in (("probe", P0, uc["thresholds"]["kidney"]), ("stack", out, tk)):
        by = {}
        for u, s in zip(units, S):
            if u[2] == "train":
                by.setdefault(u[0], [None, None])[u[1]] = s
        hit = n = 0
        for cid, v in by.items():
            yl = lab.get(cid, {}).get("kid", [None, None])
            if None in v or None in yl or 1 not in yl:
                continue
            truth = (yl[0] == 1, yl[1] == 1)
            v = np.array(v)
            pred = tuple(v >= t) if (v >= t).any() else (v[0] >= v[1], v[1] > v[0])
            hit += pred == truth
            n += 1
        st["kidney_stack"][f"train_oof_side_exact_{nm}"] = {"n": n, "acc": hit / max(n, 1)}
    pools = {"pos": sorted(round(float(v), 2) for v, y in zip(out[tr], Y[tr]) if y == 1 and v >= tk),
             "neg": sorted(round(float(v), 2) for v, y in zip(out[tr], Y[tr]) if y == 0 and v < tk)}
    uc3 = dict(uc)
    uc3["thresholds"] = dict(uc["thresholds"], kidney=tk)
    uc3["pools"] = dict(uc["pools"], kidney=pools)
    uc3["kidney_score"] = "stack(probe, flcc) v3"
    (pout / "unit_cue_v3.json").write_text(json.dumps(uc3))
    with open(pout / "stack_kidney.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "kid_right", "kid_left"])
        for cid, v in newkid.items():
            w.writerow([cid, sc[cid]["split"]] + ["" if x is None else x for x in v])

    # ---------------------------------------------------------------- 3. vertebral geometry threshold (train labels only)
    ly, lx = [], []
    for cid, l_ in lab.items():
        e = ext.get(cid)
        if not e:
            continue
        for k, lv in enumerate(UNIT_LEVELS):
            y, x = l_["vert"][k], fnum(e.get(f"{lv}_g15_loss_m"))
            if y is not None and x is not None:
                ly.append(y)
                lx.append(x)
    t_loss, f1v, prec, rec = f1_threshold(ly, lx, grid=np.round(np.arange(0.05, 0.60, 0.01), 2))
    st["vertebra_geometry"] = {"train_levels_labelled": len(ly), "train_pos": int(sum(ly)), "auroc_loss_m": auroc(ly, lx),
                               "t_loss_f1_train": t_loss, "f1": f1v, "precision": prec, "recall": rec}

    def geo_line(cid):
        e = ext.get(cid)
        if not e:
            return GEO_PREFIX + "unknown."
        vals = [(lv, fnum(e.get(f"{lv}_g15_loss_m")), fnum(e.get(f"{lv}_g15_wedge"))) for lv in UNIT_LEVELS]
        if all(v is None for _, v, _ in vals):
            return GEO_PREFIX + "unknown."
        hi = sorted([x for x in vals if x[1] is not None and x[1] >= t_loss], key=lambda x: -x[1])[:a.max_levels]
        if not hi:
            return GEO_PREFIX + f"no T8-L5 level with height loss >= {round(100 * t_loss)}%."
        return GEO_PREFIX + "; ".join(f"{lv} height loss {round(100 * v)}%" + ("" if w is None else
                                      f", anterior/posterior height ratio {w:.2f}") for lv, v, w in hi) + "."

    def bmd_item(cid):
        e = ext.get(cid, {})
        v, lv = fnum(e.get("bmd_hu")), e.get("bmd_level") or ""
        return f"{lv} trabecular bone density {round(v)} HU" if v is not None and lv else "trabecular bone density unknown"

    # ---------------------------------------------------------------- write rows
    cnt = {}
    ids600 = set(l.strip() for l in open(a.ids) if l.strip())
    v600 = []
    for sp in ("train", "val"):
        c = cnt.setdefault(sp, {"rows": 0, "kid_changed": 0, "geo_listed": 0, "geo_unknown": 0, "bmd_unknown": 0,
                                "no_ext": 0})
        with open(Path(a.inp) / f"{sp}.jsonl") as fi, open(Path(a.out) / f"{sp}.jsonl", "w") as fo:
            for line in fi:
                r = json.loads(line)
                cid = r["case_id"]
                assert r["split"] == sp
                us = r["unit_scores"]
                kid = newkid.get(cid, [None, None]) if us["kid"] is not None else None
                if kid is not None:
                    kid = [None if o is None else n for o, n in zip(us["kid"], kid)]
                c["kid_changed"] += kid != us["kid"]
                r["unit_scores"] = {"kid": kid, "vert": us["vert"]}
                pr = replace_unit_line(r["prompt"], render_unit_line(kid, us["vert"]))
                lines = pr.split("\n")
                im = [i for i, ln in enumerate(lines) if ln.startswith(MEAS_PREFIX)]
                assert len(im) == 1, cid
                i = im[0]
                assert lines[i].endswith(".")
                b = bmd_item(cid)
                lines[i] = lines[i][:-1] + "; " + b + "."
                g = geo_line(cid)
                lines.insert(i + 1, g)
                r["prompt"] = "\n".join(lines)
                assert sum(ln.startswith(UNIT_PREFIX) for ln in lines) == 1
                r["m4ext_v3"] = True
                c["rows"] += 1
                c["no_ext"] += cid not in ext
                c["bmd_unknown"] += b.endswith("unknown")
                c["geo_unknown"] += g.endswith("unknown.")
                c["geo_listed"] += "height loss" in g and not g.startswith(GEO_PREFIX + "no ")
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                if sp == "val" and cid in ids600:
                    v600.append(r)
    Path(a.val600_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.val600_out, "w") as f:
        for r in v600:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["rows"] = cnt
    st["val600_rows"] = len(v600)
    # val numbers: REPORTED only, read after every choice above was made (never used for any choice)
    yv = val_kidney_labels()
    vy = np.array([yv.get((u[0], u[1]), np.nan) for u in units])
    st["kidney_stack"]["val_auroc_probe_REPORT"] = auroc(vy[va], P0[va])
    st["kidney_stack"]["val_auroc_stack_REPORT"] = auroc(vy[va], out[va])
    st["note"] = "stack, thresholds, pools fitted / chosen on train only; val AUROC is report-only"
    (Path(a.out) / "build_v3_stats.json").write_text(json.dumps(st, indent=1, default=float))
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
