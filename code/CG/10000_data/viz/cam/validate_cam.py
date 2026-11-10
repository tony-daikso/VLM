"""Sanity / localisation check of the CAM maps against the report (disease14_qwen.jsonl quotes / detail).

Per case: unit ranking (organ mean of the map), unit of the voxel maximum, unit holding most of the top-1 % voxels,
and a disease-specific within-organ check:
  fracture      reported vertebral level(s) (regex on quotes) vs top unit; within the reported vertebra the
                fraction of its top-10 % voxels in the anterior 60 % (vertebral body) of its AP extent
  hydronephrosis reported side vs higher-scoring kidney; HU of the top-10 % CAM voxels vs the whole kidney and
                their distance to the renal hilum side (medial fraction)
  AAA           z of the CAM maximum (slice-mean profile) vs z of the largest aortic cross-section; infrarenal?
                Spearman(slice-mean CAM, slice aortic area)
  gallstones    HU of top-10 % CAM voxels vs rest of the gallbladder; Spearman(CAM, HU) inside the gallbladder
Writes <out>/cam_validation.csv and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam import DIS_ZH, bone_units_volume  # noqa: E402

VERT = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


def levels_from_quotes(qs):
    out = set()
    for q in qs:
        for m in re.finditer(r"\b([TL])\s?(\d{1,2})\b", q):
            out.add(f"{m.group(1)}{int(m.group(2))}")
    return sorted(out & set(VERT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    lab = {}
    for l in open(a.labels):
        r = json.loads(l)
        lab[r["study_id"]] = r["answer"]
    rows = []
    for f, zh in DIS_ZH.items():
        d = out / zh
        if not d.exists():
            continue
        for cd in sorted(d.iterdir()):
            mp = cd / "cam_meta.json"
            if not mp.exists():
                continue
            m = json.load(open(mp))
            cid = cd.name
            ans = lab[cid][f]
            cam = np.asanyarray(nib.load(str(cd / "cam_heatmap.nii.gz")).dataobj).transpose(2, 1, 0)
            prep = np.load(cache / "prep" / f"{cid}.npz")
            hu, aff, um = prep["hu"].astype(np.float32), prep["affine"], prep["units"]
            pres = [u for u in m["units"] if u.get("present")]
            rank = sorted(pres, key=lambda u: -u["score_voxelmap_mean"])
            if f == "fracture":
                bv, bn = bone_units_volume(cache, cid, prep["hu"], aff)
                uvol = {u["unit"]: bv == bn.index(u["unit"]) + 1 for u in pres}
            else:
                names = list(prep["unit_names"])
                uvol = {u["unit"]: um == names.index(u["unit"]) + 1 for u in pres}
            allm = np.zeros(cam.shape, bool)
            for v in uvol.values():
                allm |= v
            vals = cam[allm]
            thr = np.quantile(vals, 0.99)
            top = allm & (cam >= thr)
            share = {u: int((top & v).sum()) for u, v in uvol.items()}
            r = dict(finding=f, 疾病=zh, id=cid, quote=" | ".join(ans["quotes"][:2]), detail=json.dumps(ans["detail"]),
                     seconds=m["seconds"], top_unit_mean=rank[0]["unit"],
                     second_unit=rank[1]["unit"] if len(rank) > 1 else "",
                     unit_scores=" ".join(f"{u['unit']}:{u['score_voxelmap_mean']:.2f}" for u in rank[:4]),
                     hot_voxel_unit=m["hotspot"]["unit"], top1pct_unit=max(share, key=share.get),
                     map_vs_emb_maxabs=max(abs(u["score_voxelmap_mean"] - u["score_cached_emb"]) for u in pres))
            if f == "fracture":
                lv = levels_from_quotes(ans["quotes"])
                r["reported_levels"] = ",".join(lv)
                vr = [u for u in rank if u["unit"] in VERT]
                r["top_vertebra"] = vr[0]["unit"]
                r["unit_match"] = bool(lv) and vr[0]["unit"] in lv
                r["unit_match_pm1"] = bool(lv) and any(abs(VERT.index(vr[0]["unit"]) - VERT.index(x)) <= 1 for x in lv)
                p99 = {u: float(np.percentile(cam[uvol[u]], 99)) for u in VERT if u in uvol}
                r["top_vertebra_p99"] = max(p99, key=p99.get)
                r["p99_match"] = bool(lv) and r["top_vertebra_p99"] in lv
                if lv:
                    v = uvol[lv[0]]
                    idx = np.argwhere(v)
                    # nnU-Net order (z,y,x) of canonical RAS -> axis 1 = y (posterior->anterior)
                    y = idx[:, 1]
                    c = cam[v]
                    t10 = c >= np.quantile(c, 0.9)
                    ant = y >= y.min() + 0.4 * (y.max() - y.min())      # anterior 60 % of AP extent = body
                    r["body_frac_top10"] = round(float(ant[t10].mean()), 2)
                    r["body_frac_all"] = round(float(ant.mean()), 2)
            elif f == "hydronephrosis":
                side = ans["detail"].get("side")
                r["reported_side"] = side
                ks = {u["unit"]: u["score_voxelmap_mean"] for u in pres}
                if len(ks) == 2:
                    r["higher_kidney"] = max(ks, key=ks.get)
                    r["side_match"] = side in ("left", "right") and r["higher_kidney"] == f"kidney_{side}"
                kk = f"kidney_{side}" if side in ("left", "right") else rank[0]["unit"]
                if kk in uvol:
                    v = uvol[kk]
                    c, h = cam[v], hu[v]
                    t10 = c >= np.quantile(c, 0.9)
                    r["HU_top10"] = round(float(h[t10].mean()), 1)
                    r["HU_kidney"] = round(float(h.mean()), 1)
                    idx = np.argwhere(v)
                    x = idx[:, 2]                                      # axis 2 = x (R->L... canonical RAS: +x = left)
                    # medial = towards the midline (aorta x); midline ~ centre of body = image centre of mass of hu>-500
                    names = list(prep["unit_names"])
                    mid = np.argwhere((um == names.index("lumbar_spine") + 1) | (um == names.index("aorta") + 1))[:, 2].mean()
                    med = np.abs(x - mid) <= np.median(np.abs(x - mid))
                    r["medial_frac_top10"] = round(float(med[t10].mean()), 2)
            elif f == "abdominal_aortic_aneurysm":
                v = uvol["aorta"]
                zs = np.unique(np.argwhere(v)[:, 0])
                area = np.array([v[z].sum() for z in zs]) * 9.0         # mm^2 (3 mm grid)
                cm = np.array([cam[z][v[z]].mean() for z in zs])
                cm = np.convolve(np.pad(cm, 2, mode="edge"), np.ones(5) / 5, mode="valid")   # 15 mm smoothing
                # restrict to abdominal aorta: slices at/below the kidney top
                kid = (um == list(prep["unit_names"]).index("kidney_left") + 1) | (um == list(prep["unit_names"]).index("kidney_right") + 1)
                kz = np.argwhere(kid)[:, 0]
                zc = zs <= (kz.max() if len(kz) else zs.max())
                za, zcm = zs[zc][np.argmax(area[zc])], zs[zc][np.argmax(cm[zc])]   # both within abdominal aorta
                r["z_cam_max_whole_aorta"] = int(zs[np.argmax(cm)])
                r["z_cam_max"] = int(zcm)
                r["z_maxarea_abd"] = int(za)
                r["dz_mm"] = float(abs(int(zcm) - int(za)) * 3)
                r["cam_peak_infrarenal"] = bool(len(kz) and zcm <= np.median(kz))   # below kidney mid-level (hilum)
                r["spearman_slice_cam_area_abd"] = round(float(spearmanr(cm[zc], area[zc]).correlation), 2)
                r["max_diam_est_cm"] = round(float(2 * np.sqrt(area[zc].max() / np.pi) / 10), 1)
            elif f == "gallstones":
                v = uvol["gallbladder"]
                c, h = cam[v], hu[v]
                t10 = c >= np.quantile(c, 0.9)
                r["HU_top10"] = round(float(h[t10].mean()), 1)
                r["HU_rest"] = round(float(h[~t10].mean()), 1)
                r["HU_gb_max"] = float(h.max())
                r["spearman_cam_hu"] = round(float(spearmanr(c, h).correlation), 2)
                idx = np.argwhere(v)
                hi = idx[np.argmax(h)]
                hc = idx[np.argmax(c)]
                r["dist_camMax_huMax_mm"] = round(float(np.linalg.norm(hi - hc) * 3), 1)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cam_validation.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    for f, g in df.groupby("finding", sort=False):
        print("==", f)
        print(g.dropna(axis=1, how="all").drop(columns=["finding", "疾病", "detail"]).to_string(index=False))


if __name__ == "__main__":
    main()
