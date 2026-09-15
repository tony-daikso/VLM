"""
HiPaS single-slice dataset extractor.

Turns the raw HiPaS artery/vein 3D volumes (data/hipas_raw/) into the same
kind of 2D single-slice dataset we already built for CT-RATE+ReXGroundingCT:
for each of the 250 cases, pick the z-slice with the largest combined
artery+vein area, save the windowed 2D image + the 2D artery/vein masks,
and write a manifest row.

Note: HiPaS cases are a completely different patient cohort from CT-RATE --
there is no report/classification-label/organ-mask alignment for these
slices. This is a vessel-segmentation-only source, meant to be used
alongside (not merged row-for-row with) data/processed_rex/manifest.jsonl.

ct_scan.zip is read directly via zipfile (not extracted to disk first) to
avoid needing another 24GB of free space on top of the zip itself.

Resumable: skips case_ids already present in manifest.jsonl.
"""
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "hipas_raw")
CT_SCAN_ZIP = os.path.join(RAW_DIR, "ct_scan.zip")
ARTERY_DIR = os.path.join(RAW_DIR, "annotation", "artery")
VEIN_DIR = os.path.join(RAW_DIR, "annotation", "vein")

OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_hipas")
IMAGES_PNG_DIR = os.path.join(OUT_DIR, "images")
IMAGES_NPY_DIR = os.path.join(OUT_DIR, "images_npy")
MASKS_DIR = os.path.join(OUT_DIR, "masks")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
FAILED_PATH = os.path.join(OUT_DIR, "manifest_failed.jsonl")

WINDOW_LEVEL, WINDOW_WIDTH = -600, 1500

for d in (IMAGES_PNG_DIR, IMAGES_NPY_DIR, MASKS_DIR):
    os.makedirs(d, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def window_to_uint8(slice2d, level=WINDOW_LEVEL, width=WINDOW_WIDTH):
    lo, hi = level - width / 2, level + width / 2
    disp = np.clip(slice2d, lo, hi)
    disp = (disp - lo) / (hi - lo) * 255.0
    return disp.astype(np.uint8)


def already_done_ids():
    done = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["case_id"])
    return done


def load_npz_from_zip(zf, name):
    with zf.open(name) as f:
        buf = io.BytesIO(f.read())
    return np.load(buf)["data"]


def process_case(case_id, zf):
    ct_name = f"ct_scan/{case_id}.npz"
    ct = load_npz_from_zip(zf, ct_name).astype(np.float32)

    artery = np.load(os.path.join(ARTERY_DIR, f"{case_id}.npz"))["data"]
    vein = np.load(os.path.join(VEIN_DIR, f"{case_id}.npz"))["data"]

    if not (ct.shape == artery.shape == vein.shape):
        raise ValueError(f"shape mismatch: ct={ct.shape} artery={artery.shape} vein={vein.shape}")

    combined_area = (artery > 0).sum(axis=(0, 1)) + (vein > 0).sum(axis=(0, 1))
    z = int(np.argmax(combined_area))
    artery_px = int((artery[:, :, z] > 0).sum())
    vein_px = int((vein[:, :, z] > 0).sum())
    if artery_px == 0 and vein_px == 0:
        raise ValueError("no vessel pixels on any slice")

    slice_img = ct[:, :, z]
    artery2d = (artery[:, :, z] > 0).astype(np.uint8)
    vein2d = (vein[:, :, z] > 0).astype(np.uint8)

    npy_rel = os.path.join("images_npy", f"{case_id}.npy")
    png_rel = os.path.join("images", f"{case_id}.png")
    mask_rel = os.path.join("masks", f"{case_id}.npz")

    np.save(os.path.join(OUT_DIR, npy_rel), slice_img)
    png_arr = window_to_uint8(slice_img)
    Image.fromarray(np.rot90(png_arr)).save(os.path.join(OUT_DIR, png_rel))
    np.savez_compressed(os.path.join(OUT_DIR, mask_rel), artery=artery2d, vein=vein2d)

    return {
        "case_id": case_id,
        "slice_index": z,
        "grid_shape": list(ct.shape),
        "selection_method": "hipas_vessel_area_argmax",
        "image_npy": npy_rel,
        "image_png": png_rel,
        "vessel_mask_npz": mask_rel,
        "artery_pixel_count": artery_px,
        "vein_pixel_count": vein_px,
    }


def main():
    done = already_done_ids()
    log(f"{len(done)} cases already in manifest, will be skipped")

    with zipfile.ZipFile(CT_SCAN_ZIP) as zf:
        names = {n for n in zf.namelist() if n.startswith("ct_scan/") and n.endswith(".npz")}
        case_ids = sorted(n[len("ct_scan/") : -len(".npz")] for n in names)
        log(f"{len(case_ids)} cases found in {CT_SCAN_ZIP}")

        manifest_f = open(MANIFEST_PATH, "a")
        failed_f = open(FAILED_PATH, "a")
        n_ok, n_fail = 0, 0
        try:
            for i, case_id in enumerate(case_ids):
                if case_id in done:
                    continue
                try:
                    record = process_case(case_id, zf)
                    manifest_f.write(json.dumps(record) + "\n")
                    manifest_f.flush()
                    n_ok += 1
                    log(
                        f"[{i+1}/{len(case_ids)}] {case_id}: slice {record['slice_index']} "
                        f"(artery_px={record['artery_pixel_count']} vein_px={record['vein_pixel_count']})"
                    )
                except Exception as e:
                    n_fail += 1
                    failed_f.write(json.dumps({"case_id": case_id, "error": str(e)}) + "\n")
                    failed_f.flush()
                    log(f"FAILED {case_id}: {e}")
        finally:
            manifest_f.close()
            failed_f.close()

    log(f"Done. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
