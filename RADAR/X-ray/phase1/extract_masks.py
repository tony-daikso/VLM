"""
RADAR X-ray Phase 1 mask extractor.

Turns CheXmask's OriginalResolution/Padchest.csv (per-image RLE masks for
Left Lung / Right Lung / Heart, at each image's native resolution) into a
single-channel label-map PNG per PA/AP image, resized to 512x512 with
nearest-neighbor interpolation (no crop/pad -- must stay pixel-aligned with
whatever plain resize Phase 4's image pipeline uses).

Label values: 0=background, 1=left lung, 2=right lung, 3=heart. Painted in
that order (heart last) so heart wins at any lung/heart mask overlap.

Resumable: skips image_ids already present in manifest.jsonl. Failures are
logged to manifest_failed.jsonl and don't stop the run.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle_utils import get_mask_from_rle

ROOT_DIR = "/datadrive/VLM/data/X-ray/PadChest-Origin"
IDS_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "padchest_pa_ap_ids.txt")
CSV_PATH = os.path.join(ROOT_DIR, "chexmask_raw", "Padchest.csv")
OUT_DIR = os.path.join(ROOT_DIR, "processed_chexmask")
MASKS_DIR = os.path.join(OUT_DIR, "masks")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
FAILED_PATH = os.path.join(OUT_DIR, "manifest_failed.jsonl")

OUT_SIZE = (512, 512)
RCA_QUALITY_THRESHOLD = 0.7

csv.field_size_limit(sys.maxsize)


def log(msg):
    print(msg, flush=True)


def load_our_ids():
    with open(IDS_PATH) as f:
        return set(line.strip() for line in f if line.strip())


def already_done_ids():
    done = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["image_id"])
    return done


def build_label_map(row):
    height, width = int(row["Height"]), int(row["Width"])
    left_lung = get_mask_from_rle(row["Left Lung"], height, width)
    right_lung = get_mask_from_rle(row["Right Lung"], height, width)
    heart = get_mask_from_rle(row["Heart"], height, width)

    label_map = np.zeros((height, width), dtype=np.uint8)
    left_area, right_area, heart_area = left_lung.sum(), right_lung.sum(), heart.sum()

    overlap_pixels = 0
    for value, mask in ((1, left_lung), (2, right_lung), (3, heart)):
        overlap_pixels += int(((label_map > 0) & (mask > 0)).sum())
        label_map[mask > 0] = value

    total_fg = int((label_map > 0).sum())
    overlap_frac = overlap_pixels / total_fg if total_fg else 0.0
    return label_map, overlap_frac, int(left_area), int(right_area), int(heart_area)


def process_row(row):
    image_id = row["ImageID"]
    label_map, overlap_frac, left_area, right_area, heart_area = build_label_map(row)

    resized = Image.fromarray(label_map).resize(OUT_SIZE, Image.NEAREST)
    out_path = os.path.join(MASKS_DIR, image_id)
    resized.save(out_path)

    rca_mean = float(row["Dice RCA (Mean)"])
    rca_max = float(row["Dice RCA (Max)"])
    return {
        "image_id": image_id,
        "mask_relpath": os.path.join("masks", image_id),
        "dice_rca_mean": rca_mean,
        "dice_rca_max": rca_max,
        "quality_ok": rca_mean > RCA_QUALITY_THRESHOLD,
        "orig_height": int(row["Height"]),
        "orig_width": int(row["Width"]),
        "overlap_pixel_frac": overlap_frac,
        "left_lung_pixels": left_area,
        "right_lung_pixels": right_area,
        "heart_pixels": heart_area,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N matching rows (for testing)")
    args = parser.parse_args()

    os.makedirs(MASKS_DIR, exist_ok=True)

    our_ids = load_our_ids()
    log(f"{len(our_ids)} PA/AP images in our tar")

    done = already_done_ids()
    log(f"{len(done)} already in manifest, will be skipped")

    manifest_f = open(MANIFEST_PATH, "a")
    failed_f = open(FAILED_PATH, "a")
    n_ok, n_fail, n_seen = 0, 0, 0
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageID"]
                if image_id not in our_ids or image_id in done:
                    continue

                n_seen += 1
                try:
                    record = process_row(row)
                    manifest_f.write(json.dumps(record) + "\n")
                    manifest_f.flush()
                    n_ok += 1
                    if n_ok % 1000 == 0:
                        log(f"[{n_seen}] {n_ok} ok, {n_fail} failed so far")
                except Exception as e:
                    n_fail += 1
                    failed_f.write(json.dumps({"image_id": image_id, "error": str(e)}) + "\n")
                    failed_f.flush()
                    log(f"FAILED {image_id}: {e}")

                if args.limit and n_seen >= args.limit:
                    break
    finally:
        manifest_f.close()
        failed_f.close()

    log(f"Done. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
