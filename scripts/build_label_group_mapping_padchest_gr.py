"""
Builds a fine-grained-finding-label -> label_group (26-class taxonomy) mapping for the CXR
Stage 2.5 detection track (see U-VLM/X-ray/cxr/stage2_5/STAGE2_5_PLAN.md).

Source: data/X-ray/PadChest-GR/master_table.csv.zip, which has one row per (study, image,
fine-grained `label`) pair, each tagged with its `label_group`. Verified empirically that
every fine-grained label maps to exactly one label_group across all 8,787 rows (no
label appears under two different groups), so this is a clean many-to-one dict, not a
per-study or per-split thing -- it only needs to be built once from the full master_table
and applies to any subset of studies (including PadChest-GR-small, which doesn't ship its
own master_table.csv.zip).

grounded_reports_20240819.json's per-finding `labels` field uses this same fine-grained
vocabulary, so `label_to_label_group.json` lets Stage 2.5 assign each box a label_group
class without needing per-box annotations that don't exist in the raw data.

Writes label_to_label_group.json into every processed/ dir passed on the command line
(both the small-subset and full processed dirs use the same mapping).

跑法：python3 scripts/build_label_group_mapping_padchest_gr.py \
        data/X-ray/PadChest-GR-small/processed data/X-ray/PadChest-GR/processed
"""
import argparse
import csv
import io
import json
import os
import zipfile
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MASTER_TABLE_ZIP = os.path.join(ROOT_DIR, "data", "X-ray", "PadChest-GR", "master_table.csv.zip")


def build_mapping():
    label_to_groups = defaultdict(Counter)
    with zipfile.ZipFile(MASTER_TABLE_ZIP) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            for row in csv.DictReader(text):
                label = row["label"].strip()
                group = row["label_group"].strip()
                if label and group:
                    label_to_groups[label][group] += 1

    mapping = {}
    for label, counts in label_to_groups.items():
        if len(counts) > 1:
            print(f"warning: {label!r} maps to multiple groups {dict(counts)}, taking majority vote")
        mapping[label] = counts.most_common(1)[0][0]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_dirs", nargs="+", help="processed/ dirs to write label_to_label_group.json into")
    args = parser.parse_args()

    mapping = build_mapping()
    print(f"built mapping for {len(mapping)} fine-grained labels -> label_group")

    for processed_dir in args.processed_dirs:
        out_path = os.path.join(ROOT_DIR, processed_dir, "label_to_label_group.json")
        if not os.path.isdir(os.path.dirname(out_path)):
            print(f"skip {out_path}: dir doesn't exist")
            continue
        with open(out_path, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
