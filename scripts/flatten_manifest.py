"""Turn manifest.jsonl (one nested-JSON object per line) into human-browsable
files: a flat manifest.csv (one column per classification label) and a
reports/<volume_id>.txt per volume. Safe to re-run any time; it just
regenerates from manifest.jsonl, including partial progress mid-batch."""
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_rex")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
CSV_PATH = os.path.join(OUT_DIR, "manifest.csv")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

LABEL_COLUMNS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]


def main():
    rows = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    fieldnames = (
        ["volume_id", "slice_index", "grid_shape", "selection_method",
         "image_npy", "image_png", "segmentation_mask_npz",
         "num_findings_on_slice", "finding_texts", "report_txt"]
        + [f"label__{c}" for c in LABEL_COLUMNS]
    )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            volume_id = r["volume_id"]
            report = r.get("report_text") or {}
            report_txt_rel = os.path.join("reports", f"{volume_id}.txt")
            with open(os.path.join(OUT_DIR, report_txt_rel), "w") as rf:
                rf.write(f"Volume: {volume_id}\n")
                rf.write(f"Slice index: {r['slice_index']} (selection: {r['selection_method']})\n\n")
                for field in ("ClinicalInformation_EN", "Technique_EN", "Findings_EN", "Impressions_EN"):
                    rf.write(f"== {field} ==\n{report.get(field, '')}\n\n")
                rf.write("== Findings grounded on this slice (ReXGroundingCT) ==\n")
                for finding in r.get("findings_on_slice", []):
                    rf.write(f"- [{finding['category']}] {finding['text']} "
                              f"(entities={finding['entity_count']}, "
                              f"pixels_in_full_volume={finding['volume_pixel_count']})\n")

                labels_for_txt = r.get("classification_labels") or {}
                positive = [k for k, v in labels_for_txt.items() if v == 1]
                rf.write("\n== CT-RATE classification labels (18 pathologies) ==\n")
                rf.write(f"Positive: {', '.join(positive) if positive else '(none)'}\n")
                for k, v in labels_for_txt.items():
                    rf.write(f"  {k}: {v}\n")

            labels = r.get("classification_labels") or {}
            row_out = {
                "volume_id": volume_id,
                "slice_index": r["slice_index"],
                "grid_shape": "x".join(map(str, r["grid_shape"])),
                "selection_method": r["selection_method"],
                "image_npy": r["image_npy"],
                "image_png": r["image_png"],
                "segmentation_mask_npz": r["segmentation_mask_npz"],
                "num_findings_on_slice": len(r.get("findings_on_slice", [])),
                "finding_texts": " | ".join(fd["text"] for fd in r.get("findings_on_slice", [])),
                "report_txt": report_txt_rel,
            }
            for c in LABEL_COLUMNS:
                row_out[f"label__{c}"] = labels.get(c)
            writer.writerow(row_out)

    print(f"Wrote {CSV_PATH} ({len(rows)} rows) and {len(rows)} report files under {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
