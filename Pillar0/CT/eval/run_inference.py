"""
Run Pillar0-AbdomenCT zero-shot inference over the preprocessed abdomen CT cases.

For each case:
  1. rve.load_sample() -> raw HU volume (D, H, W)
  2. center pad/crop depth to 384 (H, W already 384 from vision-engine preprocessing)
  3. apply the 11 CT windows (10 anatomical + minmax) -> (11, D, H, W)
  4. model.extract_vision_feats({"abdomen_ct": volume}) -> normalized image embedding
  5. cosine similarity against each of the 90 finding text embeddings (projected + normalized)

Usage: python3 run_inference.py
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/Desktop/VLM/Pillar0/CT/rad-vision-engine")
import rve

CKPT_DIR = "/datadrive/VLM/Pillar0/CT/ckpt/Pillar0-AbdomenCT"
PROCESSED_DIR = "/datadrive/VLM/Pillar0/CT/processed_abdomen_1.5mm"
TEXT_EMB_PATH = "/datadrive/VLM/Pillar0/CT/text_embeddings.pt"
OUT_CSV = "/root/Desktop/VLM/Pillar0/CT/results/Pillar0_infer_results_CG2_abdomen72_v2_fixed.csv"
TARGET_SPATIAL = 384
# From src/miniclip/data_configs/pillar0_merlin_abd_ct_384.yaml -- the actual training
# preprocessing config. Applied as a final (x - mean) / std after windowing to [0, 1].
IMAGE_MEAN = 0.289
IMAGE_STD = 0.198


def pad_crop_depth(vol: torch.Tensor, target: int) -> torch.Tensor:
    """Center pad/crop the depth (dim 0) of a (D, H, W) volume to `target`, matching
    ct_processor.py's convention of padding with the volume's own min value."""
    d = vol.shape[0]
    pad_value = vol.min()
    if d == target:
        return vol
    if d < target:
        total = target - d
        before = total // 2
        after = total - before
        pad = torch.full((before, *vol.shape[1:]), pad_value, dtype=vol.dtype)
        pad2 = torch.full((after, *vol.shape[1:]), pad_value, dtype=vol.dtype)
        return torch.cat([pad, vol, pad2], dim=0)
    # crop
    start = (d - target) // 2
    return vol[start:start + target]


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print("Loading Pillar0-AbdomenCT model...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = model.cuda().eval()

    print("Loading + projecting text embeddings...")
    text_data = torch.load(TEXT_EMB_PATH)
    keys = text_data["keys"]
    raw_text_emb = text_data["embeddings"].cuda()  # (90, 4096) unnormalized
    with torch.no_grad():
        text_features = model.model.encode_text(raw_text_emb, normalize=True)  # (90, embed_dim)
    print("Text features:", text_features.shape)

    mapping_csv = os.path.join(PROCESSED_DIR, "mapping.csv")
    cases = []
    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    print(f"Found {len(cases)} processed cases")

    results = []
    with torch.no_grad():
        for i, row in enumerate(cases):
            out_path = row["output_path"]
            src_path = row["source_path"]
            # patient id = last path component of the original DICOM dir, e.g. .../abdomen/image/61 -> 61
            patient_id = os.path.basename(src_path.rstrip("/"))

            vol = rve.load_sample(out_path)  # (D, H, W) int16, raw HU
            vol = vol.float()
            vol = pad_crop_depth(vol, TARGET_SPATIAL)
            assert vol.shape == (TARGET_SPATIAL, TARGET_SPATIAL, TARGET_SPATIAL), vol.shape

            windowed = rve.apply_windowing(vol, windows="all", modality="CT")  # (11, D, H, W), range [0, 1]
            windowed = (windowed - IMAGE_MEAN) / IMAGE_STD  # match training preprocessing
            image = windowed.unsqueeze(0).cuda()  # (1, 11, D, H, W)

            image_features = model.extract_vision_feats(image={"abdomen_ct": image})  # (1, embed_dim)
            image_features = F.normalize(image_features, dim=-1)

            scores = (image_features @ text_features.T).squeeze(0).cpu().numpy()  # (90,)

            row_out = {"patient_id": patient_id}
            for k, s in zip(keys, scores):
                row_out[k] = float(s)
            results.append(row_out)
            print(f"[{i + 1}/{len(cases)}] patient {patient_id}: done")

            del image, image_features, vol, windowed
            torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"] + keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {OUT_CSV}")


if __name__ == "__main__":
    main()
