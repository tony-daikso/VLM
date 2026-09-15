"""
CT-RATE + RadGenome-ChestCT single-slice dataset extractor.

For each volume that RadGenome-ChestCT has processed (RadGenome only kept the
primary reconstruction "_1" of every scan, so this naturally dedupes the
redundant lung/mediastinal kernel pairs in CT-RATE), this script:

  1. Streams `valid_anatomy_mask.tar.gz` once (RadGenome's 197 organ/lesion
     binary masks per volume, bundled as one big tar for the whole split) and
     buffers all per-organ 3D masks for the volume currently being read.
  2. Picks one slice index per volume:
       - if any lesion-type mask (nodule/tumor/effusion/embolism/cyst) is
         non-empty, pick the z with the largest total lesion area
       - else fall back to the middle of the lung mask's z-range
       - else fall back to the volume's middle slice
  3. Downloads the matching RadGenome "preprocessed" CT volume (same voxel
     grid as the masks, so no re-alignment needed) just to pull out that one
     slice, then deletes the downloaded volume.
  4. Writes the 2D image (npy + windowed PNG), the 2D multi-organ mask
     (npz), and a manifest row joining in CT-RATE's classification labels
     and report text.

Resumable: re-running skips volume_ids already present in manifest.jsonl.
"""
import argparse
import gzip
import io
import json
import os
import sys
import tarfile
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import get_token, hf_hub_download, hf_hub_url
from PIL import Image
import requests

REPO_CT = "ibrahimhamamci/CT-RATE"
REPO_RG = "RadGenome/RadGenome-ChestCT"
ANATOMY_TAR_PATH = "dataset/valid_anatomy_mask.tar.gz"

# RadGenome's 197 organ masks include a handful of lesion-type classes, but
# most (tumor/cyst/embolism) are noisy SAT-model false positives that don't
# correspond to any of CT-RATE's 18 verified pathology labels -- trusting
# them picks clinically irrelevant slices (e.g. a stray "kidney tumor" blob
# in a scan whose actual reported finding is a lung nodule). Only use a
# lesion mask when it maps to a CT-RATE label that is *also* positive for
# this volume, so the slice we pick is tied to a chart-verified finding.
LESION_MASK_TO_LABEL = {
    "lung nodule": "Lung nodule",
    "lung effusion": "Pleural effusion",
}
LUNG_CLASSES = {"lung", "left lung", "right lung"}

# lung window for the QA/preview PNG only; the .npy keeps raw HU values
WINDOW_LEVEL, WINDOW_WIDTH = -600, 1500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
IMAGES_PNG_DIR = os.path.join(OUT_DIR, "images")
IMAGES_NPY_DIR = os.path.join(OUT_DIR, "images_npy")
MASKS_DIR = os.path.join(OUT_DIR, "masks")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
FAILED_PATH = os.path.join(OUT_DIR, "manifest_failed.jsonl")

for d in (IMAGES_PNG_DIR, IMAGES_NPY_DIR, MASKS_DIR):
    os.makedirs(d, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_lookup_tables():
    labels_path = hf_hub_download(
        REPO_CT, "dataset/multi_abnormality_labels/valid_predicted_labels.csv", repo_type="dataset"
    )
    reports_path = hf_hub_download(
        REPO_CT, "dataset/radiology_text_reports/validation_reports.csv", repo_type="dataset"
    )
    labels_df = pd.read_csv(labels_path).set_index("VolumeName")
    reports_df = pd.read_csv(reports_path).set_index("VolumeName")
    return labels_df, reports_df


def already_done_ids():
    done = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["volume_id"])
                except Exception:
                    pass
    return done


def nii_member_to_array(tar_extractfile):
    raw = tar_extractfile.read()
    decompressed = gzip.decompress(raw)
    img = nib.Nifti1Image.from_bytes(decompressed)
    return np.asarray(img.dataobj)


def select_slice(mask_buffers, depth, label_row):
    confirmed = {}
    if label_row:
        for mask_name, label_name in LESION_MASK_TO_LABEL.items():
            if label_row.get(label_name) == 1:
                arr = mask_buffers.get(mask_name)
                if arr is not None and arr.any():
                    confirmed[mask_name] = arr
    if confirmed:
        area = np.zeros(depth, dtype=np.int64)
        for arr in confirmed.values():
            area += arr.sum(axis=(0, 1))
        z = int(np.argmax(area))
        return z, "lesion_mask_argmax", sorted(confirmed.keys())

    lung_arrs = [v for k, v in mask_buffers.items() if k in LUNG_CLASSES and v.any()]
    if lung_arrs:
        union = np.zeros_like(lung_arrs[0])
        for a in lung_arrs:
            union = union | a
        zs = np.where(union.sum(axis=(0, 1)) > 0)[0]
        if len(zs):
            z = int(zs[len(zs) // 2])
            return z, "lung_middle_fallback", ["lung"]

    return depth // 2, "volume_middle_fallback", []


def volume_id_to_preprocessed_path(volume_id):
    # valid_1011_a_1 -> dataset/valid_preprocessed/valid_1011/valid_1011a/valid_1011_a_1.nii.gz
    parts = volume_id.split("_")
    pid = f"{parts[0]}_{parts[1]}"
    scan = parts[2]
    return f"dataset/valid_preprocessed/{pid}/{pid}{scan}/{volume_id}.nii.gz"


def window_to_uint8(slice2d, level=WINDOW_LEVEL, width=WINDOW_WIDTH):
    lo, hi = level - width / 2, level + width / 2
    disp = np.clip(slice2d, lo, hi)
    disp = (disp - lo) / (hi - lo) * 255.0
    return disp.astype(np.uint8)


def process_volume(volume_id, mask_buffers, shape, labels_df, reports_df):
    depth = shape[2]
    key = f"{volume_id}.nii.gz"
    label_row = labels_df.loc[key].to_dict() if key in labels_df.index else None
    report_row = reports_df.loc[key].to_dict() if key in reports_df.index else None

    z, method, used = select_slice(mask_buffers, depth, label_row)

    repo_path = volume_id_to_preprocessed_path(volume_id)
    local_path = hf_hub_download(REPO_RG, repo_path, repo_type="dataset")
    img = nib.load(local_path)
    data = np.asarray(img.dataobj).astype(np.float32)
    if data.shape != shape:
        raise ValueError(f"image/mask shape mismatch: image={data.shape} mask={shape}")
    slice_img = data[:, :, z]

    npy_rel = os.path.join("images_npy", f"{volume_id}.npy")
    png_rel = os.path.join("images", f"{volume_id}.png")
    mask_rel = os.path.join("masks", f"{volume_id}.npz")

    np.save(os.path.join(OUT_DIR, npy_rel), slice_img)
    png_arr = window_to_uint8(slice_img)
    # nibabel array is (x, y); transpose + flip for a conventional radiological view
    Image.fromarray(np.rot90(png_arr)).save(os.path.join(OUT_DIR, png_rel))

    try:
        os.remove(local_path)
    except OSError:
        pass

    names, slices = [], []
    for name, arr in mask_buffers.items():
        m = arr[:, :, z]
        if m.any():
            names.append(name)
            slices.append(m.astype(np.uint8))
    mask_stack = np.stack(slices, axis=0) if slices else np.zeros((0,) + shape[:2], dtype=np.uint8)
    np.savez_compressed(
        os.path.join(OUT_DIR, mask_rel), mask=mask_stack, organ_names=np.array(names)
    )

    return {
        "volume_id": volume_id,
        "slice_index": z,
        "grid_shape": list(shape),
        "selection_method": method,
        "selection_masks_used": used,
        "image_npy": npy_rel,
        "image_png": png_rel,
        "segmentation_mask_npz": mask_rel,
        "mask_organ_count": len(names),
        "classification_labels": label_row,
        "report_text": report_row,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N newly processed volumes")
    args = ap.parse_args()

    log("Loading CT-RATE label/report lookup tables...")
    labels_df, reports_df = load_lookup_tables()

    done = already_done_ids()
    log(f"{len(done)} volumes already in manifest, will be skipped.")

    url = hf_hub_url(REPO_RG, ANATOMY_TAR_PATH, repo_type="dataset")
    headers = {"Authorization": f"Bearer {get_token()}"}
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    tf = tarfile.open(fileobj=r.raw, mode="r|gz")

    manifest_f = open(MANIFEST_PATH, "a")
    failed_f = open(FAILED_PATH, "a")

    current_volume = None
    current_buffers = {}
    current_shape = None
    processed_count = 0

    def finalize(volume_id, buffers, shape):
        nonlocal processed_count
        if volume_id in done:
            return
        try:
            record = process_volume(volume_id, buffers, shape, labels_df, reports_df)
            manifest_f.write(json.dumps(record) + "\n")
            manifest_f.flush()
            processed_count += 1
            log(f"[{processed_count}] {volume_id}: slice {record['slice_index']} "
                f"({record['selection_method']}, {record['mask_organ_count']} organs)")
        except Exception as e:
            failed_f.write(json.dumps({"volume_id": volume_id, "error": str(e)}) + "\n")
            failed_f.flush()
            log(f"FAILED {volume_id}: {e}")
            traceback.print_exc()

    try:
        for member in tf:
            if member.isdir() or not member.name.endswith(".nii.gz"):
                continue
            # valid_anatomy_mask/seg_<volume_id>/<organ>.nii.gz
            _, folder, fname = member.name.split("/", 2)
            volume_id = folder[len("seg_"):]
            organ = fname[: -len(".nii.gz")]

            if volume_id != current_volume:
                if current_volume is not None:
                    finalize(current_volume, current_buffers, current_shape)
                    if args.limit is not None and processed_count >= args.limit:
                        break
                current_volume = volume_id
                current_buffers = {}
                current_shape = None

            if volume_id in done:
                continue  # still need to consume the stream, just skip decoding
            arr = nii_member_to_array(tf.extractfile(member))
            current_buffers[organ] = arr.astype(bool)
            if current_shape is None:
                current_shape = arr.shape
        else:
            if current_volume is not None and (args.limit is None or processed_count < args.limit):
                finalize(current_volume, current_buffers, current_shape)
    finally:
        r.close()
        manifest_f.close()
        failed_f.close()

    log(f"Done. Newly processed: {processed_count}")


if __name__ == "__main__":
    main()
