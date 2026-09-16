"""
Build a small, byte-budgeted subset of PadChest-GR for quick training smoke
tests, without waiting for (or duplicating) the full ~36GB extraction done by
extract_dataset_padchest_gr.py.

Reads studies/boxes straight from grounded_reports_20240819.json + the
official split/label_group table (reusing the helpers in
extract_dataset_padchest_gr.py), stratifies by (split, abnormal_study) so the
subset keeps roughly the same train/validation/test and normal/abnormal mix
as the full dataset, and greedily fills each stratum (using each PNG's exact
on-disk size from the zip's central directory -- compression is "store", so
no decompression is needed just to know sizes) until the target byte budget
is hit.

classification_labels uses the *full* dataset's label_group list (not one
recomputed from the subset), so label indexing stays identical between
PadChest-GR and PadChest-GR-small -- a Stage 2 model can be pointed at either
manifest.jsonl without changing code. Because of that, and because it's a
random subsample, some rare classes (e.g. osteopenia, goiter) may end up
with very few or zero positive examples in the small set -- the per-class
summary printed at the end says so explicitly, this is not silently hidden.

Output layout mirrors data/X-ray/PadChest-GR/processed/:
  data/X-ray/<name>/processed/{manifest.jsonl, manifest_failed.jsonl,
  label_groups.json, images/}
"""
import argparse
import json
import os
import random
from collections import Counter

from PIL import Image

from extract_dataset_padchest_gr import (
    GROUNDED_JSON_PATH,
    clean_finding,
    load_master_table,
    log,
    open_padchest_zip,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_enriched_studies(master_table):
    with open(GROUNDED_JSON_PATH) as f:
        studies = json.load(f)

    enriched = []
    for study in studies:
        master_row = master_table.get(study["StudyID"])
        if master_row is None:
            continue
        findings = [clean_finding(f) for f in study.get("findings", [])]
        enriched.append(
            {
                "study": study,
                "master_row": master_row,
                "findings": findings,
                "abnormal": any(f["abnormal"] for f in findings),
            }
        )
    return enriched


def select_subset(enriched, target_bytes, seed):
    """Stratified-by-(split, abnormal) greedy fill to a total byte budget."""
    strata = {}
    for item in enriched:
        key = (item["master_row"]["split"], item["abnormal"])
        strata.setdefault(key, []).append(item)

    total_studies = len(enriched)
    rng = random.Random(seed)
    selected = []
    for items in strata.values():
        rng.shuffle(items)
        stratum_budget = target_bytes * len(items) / total_studies
        acc = 0
        for item in items:
            if acc >= stratum_budget:
                break
            selected.append(item)
            acc += item["size_bytes"]
    return selected


def build(name, target_gb, seed):
    out_dir = os.path.join(ROOT_DIR, "data", "X-ray", name, "processed")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    log("loading grounded_reports_20240819.json + master_table.csv.zip ...")
    master_table = load_master_table()
    all_label_groups = sorted({lg for entry in master_table.values() for lg in entry["label_groups"]})
    enriched = load_enriched_studies(master_table)
    log(f"{len(enriched)} studies available")

    with open_padchest_zip() as zf:
        name_by_basename = {os.path.basename(n): n for n in zf.namelist() if n.lower().endswith(".png")}
        for item in enriched:
            entry = name_by_basename.get(item["study"]["ImageID"])
            item["zip_entry"] = entry
            item["size_bytes"] = zf.getinfo(entry).file_size if entry else 0
        enriched = [it for it in enriched if it["zip_entry"]]

        selected = select_subset(enriched, target_gb * 1e9, seed)
        selected_gb = sum(it["size_bytes"] for it in selected) / 1e9
        log(f"selected {len(selected)}/{len(enriched)} studies, ~{selected_gb:.2f} GB (target {target_gb:.1f} GB)")

        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        failed_path = os.path.join(out_dir, "manifest_failed.jsonl")
        n_ok, n_fail = 0, 0
        with open(manifest_path, "w") as manifest_f, open(failed_path, "w") as failed_f:
            for item in selected:
                study_id = item["study"]["StudyID"]
                image_id = item["study"]["ImageID"]
                out_path = os.path.join(images_dir, image_id)
                try:
                    if not os.path.exists(out_path):
                        with open(out_path, "wb") as img_f:
                            img_f.write(zf.read(item["zip_entry"]))
                        Image.open(out_path).load()  # integrity check
                    present = item["master_row"]["label_groups"]
                    record = {
                        "study_id": study_id,
                        "image_id": image_id,
                        "image_relpath": os.path.join("images", image_id),
                        "split": item["master_row"]["split"],
                        "patient_id": item["master_row"]["patient_id"],
                        "findings": item["findings"],
                        "classification_labels": {lg: int(lg in present) for lg in all_label_groups},
                        "report_text": " ".join(f["sentence_en"] for f in item["findings"] if f["sentence_en"]),
                        "abnormal_study": item["abnormal"],
                    }
                    manifest_f.write(json.dumps(record) + "\n")
                    n_ok += 1
                except Exception as e:
                    failed_f.write(json.dumps({"study_id": study_id, "error": str(e)}) + "\n")
                    n_fail += 1
                    log(f"FAILED {study_id}: {e}")

    with open(os.path.join(out_dir, "label_groups.json"), "w") as f:
        json.dump(all_label_groups, f, indent=2)

    split_counts = Counter(it["master_row"]["split"] for it in selected)
    label_counts = Counter(lg for it in selected for lg in it["master_row"]["label_groups"])
    log(f"Done. ok={n_ok} fail={n_fail}")
    log(f"split counts: {dict(split_counts)}")
    log("label_group counts (subset vs full):")
    full_label_counts = Counter(lg for entry in master_table.values() for lg in entry["label_groups"])
    for lg in all_label_groups:
        flag = "  <-- 0 in subset!" if label_counts.get(lg, 0) == 0 else ""
        log(f"  {lg}: {label_counts.get(lg, 0)} / {full_label_counts[lg]}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="PadChest-GR-small", help="output dataset name under data/X-ray/")
    parser.add_argument("--target-gb", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.name, args.target_gb, args.seed)
