"""
RADAR X-ray Phase 2: turn PadChest's structured Labels/Localizations columns
into per-region captions + abnormal flags (replacing the CT track's 3-stage
LLM report-parsing pipeline -- see RADAR/X-ray/PLAN.md Phase 2).

Input: data/X-ray/PadChest-Origin/other/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
Output: data/X-ray/PadChest-Origin/processed_captions/manifest.jsonl, one
record per PA/AP image that's also in our own tar's id list
(processed_chexmask/padchest_pa_ap_ids.txt), of the form:

{
  "image_id": "...",
  "labels_raw": ["cardiomegaly", "aortic elongation"],
  "whole_image_caption": "cardiomegaly. aortic elongation.",
  "regions": {
    "left_lung":  {"caption": "normal.", "abnormal": false},
    "right_lung": {"caption": "normal.", "abnormal": false},
    "heart":      {"caption": "cardiomegaly. aortic elongation.", "abnormal": true}
  }
}

Region-assignment rules live in region_rules.py. A finding with no
identifiable region (per those rules) still counts toward
whole_image_caption but not toward any per-region caption -- see
RADAR/X-ray/PLAN.md Phase 2 ("沒有 location 的 finding 當作 whole-image
finding").
"""
import ast
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_rules import locs_to_regions, label_implies_heart

ROOT_DIR = "/datadrive/VLM/data/X-ray/PadChest-Origin"
LABELS_CSV_GZ = os.path.join(ROOT_DIR, "other", "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
IDS_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "padchest_pa_ap_ids.txt")
OUT_DIR = os.path.join(ROOT_DIR, "processed_captions")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
FAILED_PATH = os.path.join(OUT_DIR, "manifest_failed.jsonl")

REGIONS = ("left_lung", "right_lung", "heart")


def log(msg):
    print(msg, flush=True)


def load_our_ids():
    with open(IDS_PATH) as f:
        return set(line.strip() for line in f if line.strip())


def parse_list_field(raw):
    """PadChest stores Python-repr'd lists as plain strings, e.g. "['a', 'b']"
    or "[]". Normalizes (strips) every element; empty/unparseable -> []."""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def make_caption(findings):
    if not findings:
        return "normal.", False
    seen = []
    for f in findings:
        if f not in seen:
            seen.append(f)
    caption = ". ".join(seen)
    if not caption.endswith("."):
        caption += "."
    return caption, True


def build_record(row):
    image_id = row["ImageID"]
    labels_raw = parse_list_field(row["Labels"])
    sentence_groups_raw = row.get("LabelsLocalizationsBySentence", "")

    if not labels_raw or labels_raw == ["normal"]:
        regions = {r: {"caption": "normal.", "abnormal": False} for r in REGIONS}
        return {
            "image_id": image_id,
            "labels_raw": labels_raw,
            "whole_image_caption": "normal.",
            "regions": regions,
        }

    region_findings = {r: [] for r in REGIONS}

    sentence_groups = []
    try:
        parsed = ast.literal_eval(sentence_groups_raw) if sentence_groups_raw else []
        if isinstance(parsed, list):
            sentence_groups = parsed
    except Exception:
        pass

    for group in sentence_groups:
        if not isinstance(group, list):
            continue
        tokens = [str(t).strip() for t in group if str(t).strip()]
        loc_tokens = [t.lower() for t in tokens if t.lower().startswith("loc ")]
        # "normal" shows up as a literal token in sentences like
        # ['normal', 'loc cardiac', 'loc mediastinum'] -- PadChest's way of
        # saying "this region is normal". It's not a finding, so drop it
        # rather than caption the region "normal." while marking it abnormal.
        label_tokens = [
            t for t in tokens
            if not t.lower().startswith("loc ") and t.lower() != "normal"
        ]
        if not label_tokens:
            continue

        region_hits = locs_to_regions(loc_tokens)
        for label in label_tokens:
            if label_implies_heart(label):
                region_hits.add("heart")

        for region in region_hits:
            region_findings[region].extend(label_tokens)

    regions = {}
    for r in REGIONS:
        caption, abnormal = make_caption(region_findings[r])
        regions[r] = {"caption": caption, "abnormal": abnormal}

    whole_caption, _ = make_caption(labels_raw)

    return {
        "image_id": image_id,
        "labels_raw": labels_raw,
        "whole_image_caption": whole_caption,
        "regions": regions,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    our_ids = load_our_ids()
    log(f"{len(our_ids)} PA/AP images in our tar")

    manifest_f = open(MANIFEST_PATH, "w")
    failed_f = open(FAILED_PATH, "w")
    n_ok, n_fail, n_seen = 0, 0, 0

    import csv
    csv.field_size_limit(sys.maxsize)

    with gzip.open(LABELS_CSV_GZ, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("ImageID")
            if image_id not in our_ids:
                continue
            n_seen += 1
            try:
                record = build_record(row)
                manifest_f.write(json.dumps(record) + "\n")
                n_ok += 1
                if n_ok % 10000 == 0:
                    log(f"[{n_seen}] {n_ok} ok, {n_fail} failed so far")
            except Exception as e:
                n_fail += 1
                failed_f.write(json.dumps({"image_id": image_id, "error": str(e)}) + "\n")
                log(f"FAILED {image_id}: {e}")

    manifest_f.close()
    failed_f.close()

    log(f"Done. ok={n_ok} fail={n_fail} matched_out_of_our_ids={n_seen}/{len(our_ids)}")


if __name__ == "__main__":
    main()
