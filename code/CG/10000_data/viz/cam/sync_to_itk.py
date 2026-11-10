"""Copy CAM outputs from owl_viz_cam/<疾病>/<id>/ into OWL_ITK/<疾病>/<id>/ when that folder already has ct.nii.gz.
Volumes: cam_heatmap.nii.gz, cam_ext.nii.gz, cam_dist_mm.nii.gz (same grid -> plain copy, else resampled onto ct.nii.gz,
linear); json: cam_meta.json, cam_ext_meta.json. Never touches the other files there. Idempotent.
"""
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/owl_viz_cam")
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/tony.tu/Desktop/OWL_ITK")
VOLS = ["cam_heatmap.nii.gz", "cam_ext.nii.gz", "cam_dist_mm.nii.gz"]
JSONS = ["cam_meta.json", "cam_ext_meta.json"]

n = {"copy": 0, "resample": 0, "no_itk_folder": 0}
for cd in sorted(SRC.glob("*/EX*")):
    if cd.parent.name.startswith("_"):
        continue
    d = DST / cd.parent.name / cd.name
    if not (d / "ct.nii.gz").exists():
        n["no_itk_folder"] += 1
        continue
    ct = nib.load(str(d / "ct.nii.gz"))
    for v in VOLS:
        if not (cd / v).exists():
            continue
        cm = nib.load(str(cd / v))
        if ct.shape == cm.shape and np.allclose(ct.affine, cm.affine, atol=1e-3):
            shutil.copyfile(cd / v, d / v)
            n["copy"] += 1
        else:
            from nibabel.processing import resample_from_to
            r = resample_from_to(cm, ct, order=1)
            nib.save(nib.Nifti1Image(np.asanyarray(r.dataobj).astype(np.float32), ct.affine), str(d / v))
            n["resample"] += 1
    for j in JSONS:
        if (cd / j).exists():
            shutil.copyfile(cd / j, d / j)
print(n)
