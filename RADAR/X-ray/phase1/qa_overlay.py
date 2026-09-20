"""
Sample a handful of extracted masks and overlay them on the actual PadChest
PNG (percentile-normalized 16-bit->8-bit per DATA_PREP_NOTES.md, then resized
plain 512x512 to match the mask) so a human can eyeball whether the lung/
heart regions actually land in the right place.

Output: data/X-ray/PadChest-Origin/processed_chexmask/qa/<image_id>.png
(original|overlay side by side).
"""
import argparse
import io
import json
import os
import random
import tarfile

import numpy as np
from PIL import Image

ROOT_DIR = "/datadrive/VLM/data/X-ray/PadChest-Origin"
TAR_PATH = os.path.join(ROOT_DIR, "PNG_tar", "PADCHEST_PA_AP.tar")
OUT_DIR = os.path.join(ROOT_DIR, "processed_chexmask")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
MASKS_DIR = os.path.join(OUT_DIR, "masks")
QA_DIR = os.path.join(OUT_DIR, "qa")

OUT_SIZE = (512, 512)
# label value -> RGBA overlay color (matches extract_masks.py's 1=left lung,
# 2=right lung, 3=heart convention)
COLORS = {
    1: (255, 80, 80, 110),   # left lung: red
    2: (80, 160, 255, 110),  # right lung: blue
    3: (80, 220, 120, 140),  # heart: green
}


def percentile_normalize_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    # vendored from DINO_LLM/X-ray/stage_dino_ssl/dataset.py -- see DATA_PREP_NOTES.md
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 255.0).astype(np.uint8)


def load_manifest():
    records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_overlay(image_id, tf, members_by_name):
    member = members_by_name[image_id]
    with tf.extractfile(member) as fh:
        raw = np.array(Image.open(io.BytesIO(fh.read())))
    img8 = percentile_normalize_uint8(raw)
    img_resized = Image.fromarray(img8).convert("RGB").resize(OUT_SIZE, Image.BILINEAR)

    mask = np.array(Image.open(os.path.join(MASKS_DIR, image_id)))
    overlay_rgba = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    overlay_px = np.zeros((*OUT_SIZE, 4), dtype=np.uint8)
    for value, color in COLORS.items():
        overlay_px[mask == value] = color
    overlay_rgba = Image.fromarray(overlay_px, mode="RGBA")

    base_rgba = img_resized.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay_rgba)

    side_by_side = Image.new("RGB", (OUT_SIZE[0] * 2, OUT_SIZE[1]))
    side_by_side.paste(img_resized, (0, 0))
    side_by_side.paste(composited.convert("RGB"), (OUT_SIZE[0], 0))
    return side_by_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(QA_DIR, exist_ok=True)
    records = load_manifest()
    print(f"{len(records)} records in manifest")

    random.seed(args.seed)
    sample = random.sample(records, min(args.n, len(records)))

    with tarfile.open(TAR_PATH) as tf:
        members_by_name = {os.path.basename(m.name): m for m in tf.getmembers()}
        for rec in sample:
            image_id = rec["image_id"]
            out_path = os.path.join(QA_DIR, image_id)
            side_by_side = make_overlay(image_id, tf, members_by_name)
            side_by_side.save(out_path)
            print(f"wrote {out_path} (dice_rca_mean={rec['dice_rca_mean']:.3f})")


if __name__ == "__main__":
    main()
