"""
Canonical patient-wise train/val split for the whole RADAR X-ray track.

Computed once here and saved to RADAR/X-ray/data_split.json -- every
downstream script (Phase 3's pretrain_segmentation.py, Phase 4's dataset
pipeline, Phase 5's RADAR training) should load that file rather than
re-deriving the split by re-running random.shuffle with the same seed.
Re-deriving independently in each script is fragile (depends on manifest
row order, Python/random module version, and every script's --limit/filter
logic staying in lockstep) -- a persisted file removes that risk and makes
the split auditable.

Split is patient-wise (grouped by PatientID from the raw PadChest labels
CSV), not image-wise: PadChest commonly has multiple images per patient, so
an image-wise random split would leak a patient's images across train/val.

Usage:
    cd RADAR/X-ray
    python3 make_data_split.py
"""
import csv
import gzip
import json
import os
import random
import sys

ROOT_DIR = "/root/Desktop/VLM/data/X-ray/PadChest-Origin"
MANIFEST_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "manifest.jsonl")
LABELS_CSV_GZ = "/datadrive/VLM/data/X-ray/PadChest-Origin/other/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_split.json")

SEED = 42
VAL_FRAC = 0.05


def load_patient_map():
    csv.field_size_limit(sys.maxsize)
    mapping = {}
    with gzip.open(LABELS_CSV_GZ, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["ImageID"]] = row["PatientID"]
    return mapping


def load_quality_ok_image_ids():
    ids = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("quality_ok", False):
                    ids.append(rec["image_id"])
    return ids


def main():
    random.seed(SEED)

    image_ids = load_quality_ok_image_ids()
    print(f"{len(image_ids)} quality_ok images in manifest")

    patient_of = load_patient_map()
    by_patient = {}
    for image_id in image_ids:
        pid = patient_of.get(image_id, image_id)
        by_patient.setdefault(pid, []).append(image_id)

    patient_ids = list(by_patient.keys())
    random.shuffle(patient_ids)
    target_val_count = max(1, int(len(image_ids) * VAL_FRAC))

    val_image_ids, train_image_ids = [], []
    val_patient_ids, train_patient_ids = [], []
    val_count = 0
    for pid in patient_ids:
        if val_count < target_val_count:
            val_image_ids.extend(by_patient[pid])
            val_patient_ids.append(pid)
            val_count += len(by_patient[pid])
        else:
            train_image_ids.extend(by_patient[pid])
            train_patient_ids.append(pid)

    print(f"{len(patient_ids)} unique patients -> train={len(train_image_ids)} images "
          f"({len(train_patient_ids)} patients), val={len(val_image_ids)} images ({len(val_patient_ids)} patients)")

    split = {
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "source_manifest": MANIFEST_PATH,
        "n_images_total": len(image_ids),
        "n_patients_total": len(patient_ids),
        "train_image_ids": sorted(train_image_ids),
        "val_image_ids": sorted(val_image_ids),
        "train_patient_ids": sorted(train_patient_ids),
        "val_patient_ids": sorted(val_patient_ids),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(split, f)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
