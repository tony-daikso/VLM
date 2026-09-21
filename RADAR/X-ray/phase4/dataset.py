"""
RADAR X-ray Phase 4: the dataset pipeline that replaces the CT track's
`caption_datasets.py` (NIfTI image + 3D mask + LLM-parsed organ captions)
with PNG + 2D CheXmask mask + Phase 2's rule-based region captions.

Joins three existing manifests in memory (all small enough to load whole;
no new manifest file is produced):
  - RADAR/X-ray/data_split.json          (Phase 3's canonical patient-wise split)
  - processed_chexmask/manifest.jsonl    (Phase 1: masks + quality_ok flag)
  - processed_captions/manifest.jsonl    (Phase 2: region captions + abnormal flags)

`RadarXrayDataset.__getitem__` returns exactly the dict keys
`RadarPretrain.forward()` reads (see radar_pretrain.py:141-146):
  image, seg, text_input, index, patient_id, organ_abnormal_flags
`samples['iters']` is NOT produced here -- radar_pretrain.py:153 expects the
training loop/runner to inject it per step, not the dataset.

Deliberately NOT registered with the LAVIS registry/BaseDataset (unlike CT's
caption_datasets.py) -- pulling in the full LAVIS import chain (decord,
fairscale, transformers, iopath, omegaconf) is unnecessary weight for a
plain torch Dataset, and pretrain_segmentation.py already established the
lighter direct-import pattern this file follows. If Phase 5 ends up using
the LAVIS Runner after all, wrap this class in a registry builder then --
cheap to add later, expensive to carry now.
"""
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT_DIR = "/root/Desktop/VLM/data/X-ray/PadChest-Origin"
TAR_PATH = os.path.join(ROOT_DIR, "PNG_tar", "PADCHEST_PA_AP.tar")
MASKS_TAR_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "masks.tar")
CHEXMASK_MANIFEST_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "manifest.jsonl")
CAPTIONS_MANIFEST_PATH = os.path.join(ROOT_DIR, "processed_captions", "manifest.jsonl")
SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_split.json")
LABELS_CSV_GZ = "/datadrive/VLM/data/X-ray/PadChest-Origin/other/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"

IMG_SIZE = 512
ORGANS = ["left_lung", "right_lung", "heart"]  # must match vision_branch.py/radar_pretrain.py self.organs order

# any-organ-abnormal records get this weight in RadarXrayDataset.sample_weights()
# (all-3-normal records get weight 1.0). Natural any-organ-abnormal rate in the
# train split is ~12.7%, so at batch_size=8 the anatomy-wise ITC branch saw an
# abnormal+intact sample for a given organ in well under 1 image per batch on
# average (each organ's own abnormal rate is only 6-9%, since most findings are
# localized to a single organ) -- see phase5/README.md for the derivation. A
# weight of 6.0 raises the any-organ-abnormal fraction of a weighted-sampled
# batch to ~46% in expectation (6*11408 / (6*11408 + 78772), using the train
# split's actual any-abnormal/all-normal counts), giving that branch enough
# signal per batch to actually learn from instead of being data-starved.
ABNORMAL_OVERSAMPLE_WEIGHT = 6.0


def percentile_normalize_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    # vendored from DINO_LLM/X-ray/stage_dino_ssl/dataset.py -- see DATA_PREP_NOTES.md
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 255.0).astype(np.uint8)


def load_patient_map():
    """image_id -> PatientID, from the raw PadChest labels CSV. Same logic as
    pretrain_segmentation.py's load_patient_map() -- kept in sync manually
    since this file avoids importing that script (it isn't meant as a
    library, just a standalone training entry point)."""
    import csv
    import gzip

    csv.field_size_limit(sys.maxsize)
    mapping = {}
    with gzip.open(LABELS_CSV_GZ, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["ImageID"]] = row["PatientID"]
    return mapping


def _load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["image_id"]] = rec
    return records


class RadarXrayDataset(Dataset):
    def __init__(self, split="train", flip_prob=0.2):
        self.flip_prob = flip_prob

        with open(SPLIT_PATH) as f:
            split_data = json.load(f)
        split_ids = set(split_data[f"{split}_image_ids"])

        chexmask = _load_jsonl(CHEXMASK_MANIFEST_PATH)
        captions = _load_jsonl(CAPTIONS_MANIFEST_PATH)
        patient_of = load_patient_map()

        self.records = []
        n_missing_caption = 0
        for image_id in split_ids:
            mask_rec = chexmask.get(image_id)
            if mask_rec is None or not mask_rec.get("quality_ok", False):
                continue
            cap_rec = captions.get(image_id)
            if cap_rec is None:
                n_missing_caption += 1
                continue
            self.records.append({
                "image_id": image_id,
                "patient_id": patient_of.get(image_id, image_id),
                "whole_image_caption": cap_rec["whole_image_caption"],
                "regions": cap_rec["regions"],
            })
        print(f"RadarXrayDataset[{split}]: {len(self.records)} usable records "
              f"(requested {len(split_ids)}, {n_missing_caption} missing Phase 2 captions)")

        self._tar = None
        self._members_by_name = None
        self._masks_tar = None
        self._mask_members_by_name = None

    def sample_weights(self):
        """Per-record weight for torch.utils.data.WeightedRandomSampler --
        see ABNORMAL_OVERSAMPLE_WEIGHT's docstring above."""
        return [
            ABNORMAL_OVERSAMPLE_WEIGHT if any(rec["regions"][o]["abnormal"] for o in ORGANS) else 1.0
            for rec in self.records
        ]

    def _ensure_tar_open(self):
        # tarfile handles aren't fork-safe to share across DataLoader
        # workers, so open lazily per-worker instead of in __init__ (same
        # pattern as pretrain_segmentation.py's ChexmaskSegDataset)
        if self._tar is None:
            self._tar = tarfile.open(TAR_PATH)
            self._members_by_name = {os.path.basename(m.name): m for m in self._tar.getmembers()}
        if self._masks_tar is None:
            self._masks_tar = tarfile.open(MASKS_TAR_PATH)
            self._mask_members_by_name = {os.path.basename(m.name): m for m in self._masks_tar.getmembers()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self._ensure_tar_open()
        rec = self.records[index]
        image_id = rec["image_id"]

        member = self._members_by_name[image_id]
        with self._tar.extractfile(member) as fh:
            raw = np.array(Image.open(io.BytesIO(fh.read())))
        img8 = percentile_normalize_uint8(raw)
        img = np.array(Image.fromarray(img8).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0

        mask_member = self._mask_members_by_name[image_id]
        with self._masks_tar.extractfile(mask_member) as fh:
            mask_img = Image.open(io.BytesIO(fh.read()))
            mask_img.load()
        # float, not int64: radar_pretrain.py:374 does
        # F.interpolate(seg_label.unsqueeze(1), mode='nearest') without
        # casting first, which torch refuses on integer dtypes. The CT track
        # never hit this because MONAI's LoadImaged happens to load NIfTI
        # labels as float32 by default -- values (0.0/1.0/2.0/3.0) are exact
        # in float32 so this loses no information, and dice.py's own
        # dice_loss casts back to .long() internally before indexing anyway.
        mask = np.array(mask_img, dtype=np.float32)

        # synchronized flips: image and mask must see the same flip decision
        if random.random() < self.flip_prob:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < self.flip_prob:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        image = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        seg = torch.from_numpy(mask)  # (H, W)

        text_input = {"report": rec["whole_image_caption"]}
        organ_abnormal_flags = torch.zeros(len(ORGANS), dtype=torch.bool)
        for i, organ in enumerate(ORGANS):
            region = rec["regions"][organ]
            text_input[organ] = region["caption"]
            organ_abnormal_flags[i] = region["abnormal"]

        return {
            "image": image,
            "seg": seg,
            "text_input": text_input,
            "index": index,
            "patient_id": rec["patient_id"],
            "organ_abnormal_flags": organ_abnormal_flags,
        }
