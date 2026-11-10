"""(runs on CT-rate, CPU, nice 19) NIfTI triplet per selected case x finding on the 3 mm prep grid.

spec json: list of {disease, id, heat: [29 floats or null] (model unit order), ...}
out: <out>/<disease>/<id>/{ct.nii.gz (int16 HU), units_label.nii.gz (uint8, id = model unit index+1),
                          contrib_heatmap.nii.gz (float32, unit value inside its mask, 0 outside)} + meta.json
Grid = prep/{id}.npz (canonical RAS, 3 mm, TS-identical resampling); arrays are stored there in nnU-Net (z,y,x)
order and transposed back to (x,y,z) here so the prep affine applies.
Bone units (T8..femur_right) come from the TS mask exactly as extract_bones.load_units; they overwrite main units
(only lumbar_spine overlaps = L1..L5, so label 12 is empty in the map).
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

SRC = Path.home() / "OWL/cg10k_infer"
sys.path.insert(0, str(SRC / "src"))
from extract_bones import load_units  # noqa: E402
from patches import label_lookup  # noqa: E402

OUT_MAIN = Path("/media/daikso/CTDATA/cg10k/owl/main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    cfg = {"out": OUT_MAIN, "mask_source": "totalsegmentator", "ts_task": "total",
           "organ_map": yaml.safe_load(open(SRC / "organ_map_bones.yaml"))}
    lookup = label_lookup(cfg)
    main_names = list(yaml.safe_load(open(SRC / "organ_map.yaml")).keys())   # 12, = model units[:12]
    cache = {}
    for s in spec:
        cid = s["id"]
        if cid not in cache:
            prep = np.load(OUT_MAIN / "prep" / f"{cid}.npz")
            assert list(prep["unit_names"]) == main_names
            hu, aff = prep["hu"], prep["affine"]
            _, bunits, _ = load_units(cfg, cid, lookup)
            lab = prep["units"].astype(np.uint8).copy()
            lab[bunits > 0] = bunits[bunits > 0] + len(main_names)
            cache = {cid: (hu.transpose(2, 1, 0).astype(np.int16), lab.transpose(2, 1, 0), aff)}
        hu, lab, aff = cache[cid]
        heat_u = np.array([np.nan if v is None else v for v in s["heat"]], np.float32)
        lut = np.zeros(256, np.float32)
        lut[1:len(heat_u) + 1] = np.nan_to_num(heat_u, nan=0.0)
        heat = lut[lab]
        d = Path(a.out) / s["disease_zh"] / cid
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in (("ct", hu), ("units_label", lab), ("contrib_heatmap", heat)):
            img = nib.Nifti1Image(arr, aff)
            img.set_qform(aff, 1); img.set_sform(aff, 1)
            nib.save(img, str(d / f"{name}.nii.gz"))
        # centroid (voxel index, x,y,z) of every present unit
        cents, bbox, nvox = {}, {}, {}
        for k in np.unique(lab):
            if k:
                idx = np.argwhere(lab == k)
                cents[int(k)] = [int(round(v)) for v in idx.mean(0)]
                bbox[int(k)] = [idx.min(0).tolist(), idx.max(0).tolist()]
                nvox[int(k)] = int(len(idx))
        json.dump({"shape": list(hu.shape), "affine": aff.tolist(), "centroids": cents, "bbox": bbox, "nvox": nvox,
                   "hu_min": int(hu.min()), "hu_max": int(hu.max())}, open(d / "grid.json", "w"))
        print(s["disease"], cid, hu.shape, flush=True)


if __name__ == "__main__":
    main()
