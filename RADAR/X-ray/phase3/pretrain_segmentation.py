"""
RADAR X-ray Phase 3: segmentation-pretraining for the 2D UNet.

Mirrors the CT track's `checkpoint_unet.pth` (a UNet pretrained on
TotalSegmentator masks before RADAR proper loads it) -- see
`RADAR/CT/docs/PREPROCESS.md`. There's no released X-ray equivalent, so this
script trains `VisionBranch.UNet` from scratch on CheXmask masks (Phase 1's
output) with a Dice loss only (no ITC/contrastive loss -- that only happens
once the full RadarPretrain model is assembled in Phase 5), and saves a
checkpoint in the same `{"network_weights": state_dict}` format
`vision_branch.py`'s optional checkpoint-loading code expects.

Hard dependency: RADAR/X-ray/phase1/extract_masks.py must have been run
first (data/X-ray/PadChest-Origin/processed_chexmask/manifest.jsonl must
exist and be non-empty). Images come straight from PADCHEST_PA_AP.tar,
percentile-normalized 16-bit->8-bit then resized to 512x512 (matching
DATA_PREP_NOTES.md's convention and, critically, matching the *plain
resize, no crop* the Phase 1 masks were produced with -- see
RADAR/X-ray/phase1/README.md's format contract).

This script only pretrains the segmentation head in isolation; it does not
touch the ITC/contrastive-learning code path at all (that's RadarPretrain
in radar_pretrain.py, assembled in Phase 5 once Phase 4's dataset pipeline
exists).

Usage:
    cd RADAR/X-ray/phase3
    python3 pretrain_segmentation.py [--epochs 10] [--batch-size 16] [--limit N]
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Import vision_branch.py/dice.py directly rather than through
# `lavis.models.radar_models...` -- this script doesn't use any of the
# LAVIS registry/Runner/Task machinery, and going through `lavis/__init__.py`
# would drag in unrelated heavy deps (decord, fairscale, transformers, ...)
# that only the full RADAR training env (RADAR/CT/.venv_radar/-style, see
# RADAR/CT/requirements.txt) actually needs.
_RADAR_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lavis", "models", "radar_models")
sys.path.insert(0, _RADAR_MODELS_DIR)
from vision_branch import VisionBranch
from dice import MemoryEfficientSoftDiceLoss


# Local fast disk (829MB/s), not /datadrive (fuse.fx mount, ~14MB/s write --
# far too slow for a DataLoader that re-reads every image every epoch).
# /datadrive keeps the archival originals (raw tar, CheXmask CSV); this
# script reads from local copies instead. See RADAR/X-ray/phase1/README.md.
ROOT_DIR = "/root/Desktop/VLM/data/X-ray/PadChest-Origin"
TAR_PATH = os.path.join(ROOT_DIR, "PNG_tar", "PADCHEST_PA_AP.tar")
MANIFEST_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "manifest.jsonl")
MASKS_TAR_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "masks.tar")

# Only read once at startup (17MB gz) for the image_id -> PatientID map used
# for a patient-wise train/val split -- fine to read straight off /datadrive,
# unlike the per-sample tar reads above which need the local fast copy.
LABELS_CSV_GZ = "/datadrive/VLM/data/X-ray/PadChest-Origin/other/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
CKPT_OUT_PATH = os.path.join(CKPT_DIR, "checkpoint_unet_xray.pth")

IMG_SIZE = 512
RCA_QUALITY_THRESHOLD = 0.7  # matches extract_masks.py -- drop low-quality masks from pretraining


def percentile_normalize_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    # vendored from DINO_LLM/X-ray/stage_dino_ssl/dataset.py -- see DATA_PREP_NOTES.md
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 255.0).astype(np.uint8)


def load_patient_map():
    """image_id -> PatientID, from the raw PadChest labels CSV. Used to make
    the train/val split patient-wise (a patient's images shouldn't leak
    across the split -- PadChest commonly has multiple images per patient)."""
    import csv
    import gzip

    csv.field_size_limit(sys.maxsize)
    mapping = {}
    with gzip.open(LABELS_CSV_GZ, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["ImageID"]] = row["PatientID"]
    return mapping


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} doesn't exist yet -- run RADAR/X-ray/phase1/extract_masks.py first."
        )
    records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("quality_ok", False):
                    records.append(rec)
    if not records:
        raise RuntimeError(f"{MANIFEST_PATH} has no quality_ok=True records yet.")
    return records


class ChexmaskSegDataset(Dataset):
    """(image, mask) pairs for segmentation-only pretraining. Image read
    straight from the PA/AP tar (percentile-normalize + plain resize to
    512x512, no crop); mask read from Phase 1's precomputed 512x512 PNGs,
    packed into masks.tar (a single archive rather than 96k loose files --
    see phase1/README.md) using the same lazy-per-worker tarfile pattern."""

    def __init__(self, records, tar_path=TAR_PATH, masks_tar_path=MASKS_TAR_PATH):
        self.records = records
        self.tar_path = tar_path
        self.masks_tar_path = masks_tar_path
        self._tar = None
        self._members_by_name = None
        self._masks_tar = None
        self._mask_members_by_name = None

    def _ensure_tar_open(self):
        # tarfile handles aren't fork-safe to share across DataLoader
        # workers, so open lazily per-worker instead of in __init__
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path)
            self._members_by_name = {os.path.basename(m.name): m for m in self._tar.getmembers()}
        if self._masks_tar is None:
            self._masks_tar = tarfile.open(self.masks_tar_path)
            self._mask_members_by_name = {os.path.basename(m.name): m for m in self._masks_tar.getmembers()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        self._ensure_tar_open()
        rec = self.records[idx]
        image_id = rec["image_id"]

        member = self._members_by_name[image_id]
        with self._tar.extractfile(member) as fh:
            raw = np.array(Image.open(io.BytesIO(fh.read())))
        img8 = percentile_normalize_uint8(raw)
        img = Image.fromarray(img8).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        image = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)  # (1,H,W)

        mask_member = self._mask_members_by_name[image_id]
        with self._masks_tar.extractfile(mask_member) as fh:
            mask = Image.open(io.BytesIO(fh.read()))
            mask.load()
        label = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H,W), values 0-3

        return image, label


def dice_loss_for_batch(dice_loss_fn, seg_probs, label):
    # matches radar_pretrain.py's own seg-loss computation exactly (nearest
    # -neighbor downsample of the label to segs[0]'s resolution, since the
    # light decoder's finest deep-supervision output is at input/2 res)
    target_size = seg_probs.shape[-2:]
    label_resized = F.interpolate(label.unsqueeze(1).float(), size=target_size, mode='nearest')
    return dice_loss_fn(seg_probs, label_resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N records (for a quick smoke run)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_manifest()
    print(f"{len(records)} quality_ok records in manifest")
    if args.limit:
        records = records[: args.limit]

    # Patient-wise split (PadChest commonly has multiple images per patient,
    # so an image-wise split would leak a patient across train/val). This is
    # the ONE canonical split for the whole RADAR X-ray track -- reused by
    # Phase 4/5 too -- so prefer the persisted RADAR/X-ray/data_split.json
    # (see make_data_split.py) over re-deriving it here. Re-derive only as a
    # fallback (e.g. a --limit smoke run, where the persisted full split
    # doesn't apply) and warn loudly, since an ad hoc re-derivation won't
    # match the canonical split other scripts use.
    split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_split.json")
    if os.path.exists(split_path) and not args.limit:
        with open(split_path) as f:
            split = json.load(f)
        train_ids, val_ids = set(split["train_image_ids"]), set(split["val_image_ids"])
        by_id = {rec["image_id"]: rec for rec in records}
        train_records = [by_id[i] for i in train_ids if i in by_id]
        val_records = [by_id[i] for i in val_ids if i in by_id]
        print(f"loaded canonical split from {split_path}: train={len(train_records)} val={len(val_records)}")
    else:
        warnings.warn(
            f"{split_path} not found (or --limit given) -- re-deriving an ad hoc patient-wise split "
            "that will NOT match the canonical split other phases use. Run make_data_split.py for a real run."
        )
        patient_of = load_patient_map()
        by_patient = {}
        for rec in records:
            pid = patient_of.get(rec["image_id"], rec["image_id"])  # fall back to per-image if somehow missing
            by_patient.setdefault(pid, []).append(rec)

        patient_ids = list(by_patient.keys())
        random.shuffle(patient_ids)
        target_val_count = max(1, int(len(records) * args.val_frac))

        val_records, train_records = [], []
        val_count = 0
        for pid in patient_ids:
            if val_count < target_val_count:
                val_records.extend(by_patient[pid])
                val_count += len(by_patient[pid])
            else:
                train_records.extend(by_patient[pid])
        print(f"{len(patient_ids)} unique patients -> train={len(train_records)} val={len(val_records)} (ad hoc patient-wise split)")

    train_ds = ChexmaskSegDataset(train_records)
    val_ds = ChexmaskSegDataset(val_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # VisionBranch also builds proj1/2/3 (unused here) and tries to load any
    # existing checkpoint_unet_xray.pth -- fine, that's exactly the resume
    # behavior we want for a re-run of this same script
    vb = VisionBranch().to(device)
    unet = vb.UNet

    dice_loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=None, batch_dice=False, do_bg=False, smooth=0, ddp=False)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    n_batches = len(train_loader)
    log_every = max(1, n_batches // 10)  # ~10 progress lines per epoch

    for epoch in range(args.epochs):
        unet.train()
        train_losses = []
        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            _, segs = unet(image)
            seg_probs = torch.softmax(segs[0], 1)
            loss = dice_loss_for_batch(dice_loss_fn, seg_probs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (i + 1) % log_every == 0:
                print(f"  epoch {epoch+1}/{args.epochs} batch {i+1}/{n_batches}: running_dice_loss={np.mean(train_losses[-log_every:]):.4f}")

        unet.eval()
        val_losses = []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                _, segs = unet(image)
                seg_probs = torch.softmax(segs[0], 1)
                val_losses.append(dice_loss_for_batch(dice_loss_fn, seg_probs, label).item())

        print(f"epoch {epoch+1}/{args.epochs}: train_dice_loss={np.mean(train_losses):.4f} val_dice_loss={np.mean(val_losses):.4f}")

        # save every epoch, not just at the end -- a long background run
        # shouldn't lose everything to a crash/interrupt near the finish line
        torch.save({"network_weights": unet.state_dict()}, CKPT_OUT_PATH)
        print(f"saved {CKPT_OUT_PATH} (epoch {epoch+1})")


if __name__ == "__main__":
    main()
