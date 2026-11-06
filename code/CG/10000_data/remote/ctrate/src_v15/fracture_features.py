"""Per-vertebra geometry + bone-HU features for the fracture analysis (no embeddings needed).

Inputs per case (one host's outputs dir): masks/{id}.nii.gz (TotalSegmentator 'total' 117 classes, 3 mm, saved in
RAS canonical by worker.py --engine nnunet) and prep/{id}.npz ('hu', nnU-Net (z,y,x) order). Array axes after
.transpose(2,1,0): z = craniocaudal (+ superior), y = antero-posterior (+ anterior), x = left-right.
For each level T8..L5: vertebral body = anterior 40% of the vertebra's AP extent; heights (mm) of the anterior,
middle and posterior thirds of the body (= number of z slices x spacing), body volume, body HU (median, p10).
Case-level summaries: min anterior/posterior ratio, min height / mean(neighbour heights), min volume ratio to
neighbours, L1 body HU, mean T12-L2 HU. Writes one CSV row per case.
usage: python fracture_features.py --outputs <outputs dir> --csv <file> [--shard i/n]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import nibabel as nib
import numpy as np

LEVELS = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
MIN_VOX = 150   # below: level absent / cut by the field of view (3 mm voxels, 27 mm^3 each)
SP = 3.0        # mm per voxel on the --fast grid


def level_ids():
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    return {lv: inv[f"vertebrae_{lv}"] for lv in LEVELS}


def body_feats(m, hu):
    idx = np.argwhere(m)
    if len(idx) < MIN_VOX:
        return None
    y0, y1 = idx[:, 1].min(), idx[:, 1].max()
    body = idx[idx[:, 1] >= y1 - 0.4 * (y1 - y0)]          # anterior 40% of AP extent = vertebral body
    if len(body) < 30:
        return None
    by0, by1 = body[:, 1].min(), body[:, 1].max()
    cut = np.linspace(by0, by1 + 1, 4)
    hs = []
    for a, b in zip(cut[:-1], cut[1:]):                    # posterior -> anterior thirds of the body
        part = body[(body[:, 1] >= a) & (body[:, 1] < b)]
        hs.append(len(np.unique(part[:, 0])) * SP if len(part) else np.nan)
    h_post, h_mid, h_ant = hs
    v = hu[body[:, 0], body[:, 1], body[:, 2]].astype(float)
    return {"h_ant": h_ant, "h_mid": h_mid, "h_post": h_post, "vol": len(body) * SP ** 3,
            "hu_med": float(np.median(v)), "hu_p10": float(np.percentile(v, 10)), "zc": float(body[:, 0].mean())}


def case_features(lab, hu, ids):
    per = {lv: body_feats(lab == ids[lv], hu) for lv in LEVELS}
    row = {"n_levels": sum(v is not None for v in per.values())}
    for lv in LEVELS:
        f = per[lv]
        for k in ("h_ant", "h_mid", "h_post", "vol", "hu_med"):
            row[f"{lv}_{k}"] = f[k] if f else np.nan
    ap, nb_h, nb_v = [], [], []
    for i, lv in enumerate(LEVELS):
        f = per[lv]
        if not f:
            continue
        if f["h_post"] and f["h_post"] > 0:
            ap.append(f["h_ant"] / f["h_post"])
        nbs = [per[LEVELS[j]] for j in (i - 1, i + 1) if 0 <= j < len(LEVELS) and per[LEVELS[j]]]
        if nbs:
            nb_h.append(f["h_mid"] / np.mean([n["h_mid"] for n in nbs]))
            nb_v.append(f["vol"] / np.mean([n["vol"] for n in nbs]))
    row["min_ant_post_ratio"] = min(ap) if ap else np.nan
    row["min_height_nb_ratio"] = min(nb_h) if nb_h else np.nan
    row["max_height_loss"] = 1 - min(nb_h) if nb_h else np.nan
    row["min_vol_nb_ratio"] = min(nb_v) if nb_v else np.nan
    row["L1_hu"] = per["L1"]["hu_med"] if per["L1"] else np.nan
    hus = [per[l]["hu_med"] for l in ("T12", "L1", "L2") if per[l]]
    row["T12_L2_hu"] = float(np.mean(hus)) if hus else np.nan
    row["min_body_hu_p10"] = min(per[l]["hu_p10"] for l in LEVELS if per[l]) if row["n_levels"] else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    out = Path(a.outputs)
    ids = sorted(p.stem for p in (out / "patches").glob("*.json"))
    ids = [i for i in ids if (out / "masks" / f"{i}.nii.gz").exists() and (out / "prep" / f"{i}.npz").exists()]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        ids = ids[k::n]
    lv_ids = level_ids()
    t0, rows, bad = time.time(), [], 0
    for j, cid in enumerate(ids):
        try:
            lab = np.asanyarray(nib.load(str(out / "masks" / f"{cid}.nii.gz")).dataobj).transpose(2, 1, 0)
            hu = np.load(out / "prep" / f"{cid}.npz")["hu"]
            if lab.shape != hu.shape:
                bad += 1
                continue
            rows.append({"id": cid, **case_features(lab, hu, lv_ids)})
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(cid, "ERROR", repr(e), flush=True)
        if j % 500 == 0:
            print(f"[{j}/{len(ids)}] {time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0].keys()) if rows else ["id"]
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"done {len(rows)} cases, {bad} skipped -> {a.csv}")


if __name__ == "__main__":
    main()
