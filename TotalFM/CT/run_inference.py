"""
Run TotalFM zero-shot organ-level inference over the 72 abdomen CT cases,
scored against the same 90 finding prompts. TotalFM's README recommends short
organ-specific text over full report sentences, so we reuse the short-form
prompts (finding_english_map.json), matched per-organ against the correct
organ embedding (not a whole-scan embedding).
"""
import csv
import json
import os
import warnings

warnings.filterwarnings("ignore")

from totalfm import TotalFM

NIFTI_DIR = "/datadrive/VLM/Merlin/CT/nifti"
SEG_DIR = "/datadrive/VLM/TotalFM/CT/segmentations"
FINDING_MAP_PATH = "/datadrive/VLM/Pillar0/CT/finding_english_map.json"
OUT_CSV = "/root/Desktop/VLM/TotalFM/CT/results/TotalFM_infer_results_CG2_abdomen72.csv"

# Map our Chinese organ names (via organ_en) to TotalSegmentator's organ label names.
# TotalFM keys embeddings by TotalSegmentator label names.
ORGAN_EN_TO_TS = {
    "Liver": "liver",
    "Spleen": "spleen",
    "Pancreas": "pancreas",
    "Gallbladder": "gallbladder",
    "Kidney": ["kidney_left", "kidney_right"],
    "Adrenal gland": ["adrenal_gland_left", "adrenal_gland_right"],
    "Stomach": "stomach",
    "Small bowel": "small_bowel",
    "Large bowel": "colon",
    "Duodenum": "duodenum",
    "Esophagus": "esophagus",
    "Aorta": "aorta",
    "Portal vein": "portal_vein_and_splenic_vein",
    "Heart": "heart",
    "Lung": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "Bladder": "urinary_bladder",
    "Rib": None,
    "Sacrum": "sacrum",
}


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    finding_map = json.load(open(FINDING_MAP_PATH, encoding="utf-8"))
    keys = list(finding_map.keys())

    usable_keys = [k for k in keys if ORGAN_EN_TO_TS.get(finding_map[k]["organ_en"]) is not None]
    skipped = [k for k in keys if k not in usable_keys]
    print(f"{len(usable_keys)}/{len(keys)} findings map to a TotalSegmentator organ; skipping {len(skipped)}: {skipped}")

    print("Loading TotalFM model...")
    model = TotalFM(device="cuda")

    prompts = {k: f"{finding_map[k]['organ_en']}: {finding_map[k]['finding_en']}." for k in usable_keys}
    text_embeddings = model.encode_text(list(prompts.values()))

    patient_ids = sorted(
        [f[:-len(".nii.gz")] for f in os.listdir(NIFTI_DIR) if f.endswith(".nii.gz")],
        key=lambda x: int(x),
    )
    print(f"{len(patient_ids)} patients to process")

    results = []
    for i, pid in enumerate(patient_ids):
        ct_path = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
        seg_path = os.path.join(SEG_DIR, f"{pid}.nii.gz")
        if not os.path.exists(seg_path):
            print(f"[{i + 1}/{len(patient_ids)}] patient {pid}: no segmentation, skip")
            continue

        image_embeddings = model.encode_image(ct_path=ct_path, seg_path=seg_path)

        row_out = {"patient_id": pid}
        for k in usable_keys:
            organ_en = finding_map[k]["organ_en"]
            ts_names = ORGAN_EN_TO_TS[organ_en]
            if isinstance(ts_names, str):
                ts_names = [ts_names]

            organ_emb = None
            for name in ts_names:
                if name in image_embeddings:
                    organ_emb = image_embeddings[name]
                    break
            if organ_emb is None:
                row_out[k] = ""
                continue

            text_emb = text_embeddings[prompts[k]]
            sim = float((organ_emb * text_emb).sum())
            row_out[k] = sim
        results.append(row_out)
        print(f"[{i + 1}/{len(patient_ids)}] patient {pid}: done")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"] + usable_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {OUT_CSV}")


if __name__ == "__main__":
    main()
