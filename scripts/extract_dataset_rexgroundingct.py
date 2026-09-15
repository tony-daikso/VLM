"""
CT-RATE + ReXGroundingCT single-slice dataset extractor.

Segmentation source: ReXGroundingCT instead of RadGenome-ChestCT. Unlike
RadGenome's masks (resampled to a 368x368x63 grid with the affine dropped),
ReXGroundingCT's per-finding masks are stored on the *exact same voxel grid*
as the original CT-RATE volume (verified: shape match + this is literally
what the CT-RATE image looked like when the annotators drew on it), so we
use the original high-resolution CT-RATE image directly as the `image`
field -- no resampling, no alignment risk.

Trade-off vs RadGenome: ReXGroundingCT only covers 3,142 of CT-RATE's ~50k
volumes, and only annotates *findings* (pathology regions), not normal
anatomy. Those 3,142 volumes split as: 414 from CT-RATE's validation split,
and 2,728 from CT-RATE's training split (2,578 in ReXGroundingCT's own
"train" section + 50 in its "val" + 100 in its "test") -- all 3,142 are
this script's universe, regardless of which CT-RATE split they come from.

For each of those volumes:
  1. Look up its findings/pixel-counts/shape from ReXGroundingCT's
     dataset.json (already loaded in full, no need to stream anything).
  2. Pick slice_index = the z with the largest combined area across all
     finding masks (ties broken by first).
  3. Download the original CT-RATE volume just to pull that one slice, then
     delete it. Download the (small, per-volume) ReXGroundingCT mask file,
     slice it the same way, then delete it too.
  4. Save image (npy + windowed PNG), mask (npz: per-finding 2D arrays +
     their finding text/category/pixel count), and a manifest row with
     CT-RATE's classification labels + report text joined in.

Resumable: re-running skips volume_ids already present in manifest.jsonl.
"""
import argparse
import json
import os
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_CT = "ibrahimhamamci/CT-RATE"
REPO_REX = "rajpurkarlab/ReXGroundingCT"

WINDOW_LEVEL, WINDOW_WIDTH = -600, 1500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_rex")
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
    label_paths = [
        hf_hub_download(REPO_CT, "dataset/multi_abnormality_labels/valid_predicted_labels.csv", repo_type="dataset"),
        hf_hub_download(REPO_CT, "dataset/multi_abnormality_labels/train_predicted_labels.csv", repo_type="dataset"),
    ]
    report_paths = [
        hf_hub_download(REPO_CT, "dataset/radiology_text_reports/validation_reports.csv", repo_type="dataset"),
        hf_hub_download(REPO_CT, "dataset/radiology_text_reports/train_reports.csv", repo_type="dataset"),
    ]
    labels_df = pd.concat([pd.read_csv(p) for p in label_paths]).set_index("VolumeName")
    reports_df = pd.concat([pd.read_csv(p) for p in report_paths]).set_index("VolumeName")

    rex_meta_path = hf_hub_download(REPO_REX, "dataset.json", repo_type="dataset")
    rex_meta = json.load(open(rex_meta_path))
    rex_entries = {}
    for split in ("train", "val", "test"):
        for entry in rex_meta[split]:
            name = entry["name"]
            # covers both CT-RATE splits (valid_* and train_*) -- ReXGroundingCT's
            # own train/val/test sections mix volumes from both.
            rex_entries[name[: -len(".nii.gz")]] = entry
    return labels_df, reports_df, rex_entries


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


def volume_id_to_ct_rate_path(volume_id):
    # valid_1016_c_1 -> dataset/valid_fixed/valid_1016/valid_1016_c/valid_1016_c_1.nii.gz
    # train_1741_b_2 -> dataset/train_fixed/train_1741/train_1741_b/train_1741_b_2.nii.gz
    parts = volume_id.split("_")
    split_name = parts[0]
    pid = f"{parts[0]}_{parts[1]}"
    scan = parts[2]
    return f"dataset/{split_name}_fixed/{pid}/{pid}_{scan}/{volume_id}.nii.gz"


def cleanup_download(path):
    """Remove a hf_hub_download()'d file for real.

    hf_hub_download returns a symlink into .cache/huggingface/hub/.../blobs/;
    os.remove()-ing just the symlink leaves the actual (large) blob file
    behind, silently wasting disk. Resolve to the real blob and remove both.
    """
    real = os.path.realpath(path)
    for p in {path, real}:
        try:
            os.remove(p)
        except OSError:
            pass


def window_to_uint8(slice2d, level=WINDOW_LEVEL, width=WINDOW_WIDTH):
    lo, hi = level - width / 2, level + width / 2
    disp = np.clip(slice2d, lo, hi)
    disp = (disp - lo) / (hi - lo) * 255.0
    return disp.astype(np.uint8)


def select_slice(seg):
    # seg: (F, H, W, D)
    total_area = (seg > 0).any(axis=0).sum(axis=(0, 1))
    z = int(np.argmax(total_area))
    present = [i for i in range(seg.shape[0]) if (seg[i, :, :, z] > 0).any()]
    return z, present


def process_volume(volume_id, rex_entry, labels_df, reports_df):
    ct_path = hf_hub_download(REPO_CT, volume_id_to_ct_rate_path(volume_id), repo_type="dataset")
    seg_path = hf_hub_download(REPO_REX, f"segmentations/{volume_id}.nii.gz", repo_type="dataset")

    ct_img = nib.load(ct_path)
    data = np.asarray(ct_img.dataobj).astype(np.float32)
    seg = np.asarray(nib.load(seg_path).dataobj)
    if data.shape != seg.shape[1:]:
        raise ValueError(f"image/mask shape mismatch: image={data.shape} mask={seg.shape}")

    z, present = select_slice(seg)
    slice_img = data[:, :, z]

    npy_rel = os.path.join("images_npy", f"{volume_id}.npy")
    png_rel = os.path.join("images", f"{volume_id}.png")
    mask_rel = os.path.join("masks", f"{volume_id}.npz")

    np.save(os.path.join(OUT_DIR, npy_rel), slice_img)
    png_arr = window_to_uint8(slice_img)
    Image.fromarray(np.rot90(png_arr)).save(os.path.join(OUT_DIR, png_rel))

    findings_meta = []
    slices = []
    for i in present:
        slices.append(seg[i, :, :, z])
        findings_meta.append(
            {
                "index": i,
                "text": rex_entry["findings"][str(i)],
                "category": rex_entry["categories"][str(i)],
                "entity_count": rex_entry["entity_counts"][str(i)],
                "volume_pixel_count": rex_entry["pixels"][str(i)],
            }
        )
    mask_stack = np.stack(slices, axis=0) if slices else np.zeros((0,) + data.shape[:2], dtype=np.uint8)
    np.savez_compressed(
        os.path.join(OUT_DIR, mask_rel),
        mask=mask_stack,
        finding_indices=np.array(present),
    )

    for p in (ct_path, seg_path):
        cleanup_download(p)

    key = f"{volume_id}.nii.gz"
    label_row = labels_df.loc[key].to_dict() if key in labels_df.index else None
    report_row = reports_df.loc[key].to_dict() if key in reports_df.index else None

    return {
        "volume_id": volume_id,
        "slice_index": z,
        "grid_shape": list(data.shape),
        "selection_method": "rexgroundingct_finding_argmax",
        "image_npy": npy_rel,
        "image_png": png_rel,
        "segmentation_mask_npz": mask_rel,
        "findings_on_slice": findings_meta,
        "classification_labels": label_row,
        "report_text": report_row,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N newly processed volumes")
    args = ap.parse_args()

    log("Loading lookup tables (CT-RATE labels/reports + ReXGroundingCT metadata)...")
    labels_df, reports_df, rex_entries = load_lookup_tables()
    log(f"{len(rex_entries)} CT-RATE volumes (train+valid splits) have a ReXGroundingCT mask.")

    done = already_done_ids()
    log(f"{len(done)} volumes already in manifest, will be skipped.")

    manifest_f = open(MANIFEST_PATH, "a")
    failed_f = open(FAILED_PATH, "a")
    processed_count = 0

    try:
        for volume_id, rex_entry in rex_entries.items():
            if volume_id in done:
                continue
            try:
                record = process_volume(volume_id, rex_entry, labels_df, reports_df)
                manifest_f.write(json.dumps(record) + "\n")
                manifest_f.flush()
                processed_count += 1
                log(
                    f"[{processed_count}] {volume_id}: slice {record['slice_index']} "
                    f"({len(record['findings_on_slice'])} findings)"
                )
            except Exception as e:
                failed_f.write(json.dumps({"volume_id": volume_id, "error": str(e)}) + "\n")
                failed_f.flush()
                log(f"FAILED {volume_id}: {e}")
                traceback.print_exc()
            if args.limit is not None and processed_count >= args.limit:
                break
    finally:
        manifest_f.close()
        failed_f.close()

    log(f"Done. Newly processed: {processed_count}")


if __name__ == "__main__":
    main()
