"""
Batch-add TotalSegmentator organ masks to the existing 2D single-slice
dataset in data/processed_rex/.

For each volume already in manifest.jsonl (produced by
extract_dataset_rexgroundingct.py), this script:
  1. Re-downloads the original CT-RATE volume (same path helper as the
     slice extractor).
  2. Runs TotalSegmentator (--fast, multilabel output, mps device) on the
     full 3D volume -- organ boundaries need 3D context, so we cannot run
     it on the single already-saved 2D slice.
  3. Slices the resulting organ label map at the SAME slice_index already
     recorded for that volume (so it aligns pixel-for-pixel with the
     existing image_npy / segmentation_mask_npz).
  4. Saves the 2D organ label map (uint8, TotalSegmentator "total" task
     class ids, background=0) to organ_masks/{volume_id}.npy.
  5. Deletes the downloaded CT volume + temp TotalSegmentator output.

Resumable: skips volume_ids that already have an organ_masks/{id}.npy file.
Reads manifest.jsonl once at startup, so re-run after the slice-extraction
script finishes to pick up any volumes added after this script started.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download

REPO_CT = "ibrahimhamamci/CT-RATE"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_rex")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
ORGAN_MASKS_DIR = os.path.join(OUT_DIR, "organ_masks")
LOG_PATH = os.path.join(OUT_DIR, "totalseg_organs.log")
FAILED_PATH = os.path.join(OUT_DIR, "totalseg_organs_failed.jsonl")

os.makedirs(ORGAN_MASKS_DIR, exist_ok=True)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def volume_id_to_ct_rate_path(volume_id):
    parts = volume_id.split("_")
    split_name = parts[0]
    pid = f"{parts[0]}_{parts[1]}"
    scan = parts[2]
    return f"dataset/{split_name}_fixed/{pid}/{pid}_{scan}/{volume_id}.nii.gz"


def cleanup_download(path):
    """See scripts/extract_dataset_rexgroundingct.py:cleanup_download -- hf_hub_download
    returns a symlink into .cache/huggingface/hub/.../blobs/; removing just the symlink
    leaves the real (large) blob file behind, silently wasting disk."""
    real = os.path.realpath(path)
    for p in {path, real}:
        try:
            os.remove(p)
        except OSError:
            pass


def load_manifest_records():
    records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_done_ids():
    done = set()
    for fn in os.listdir(ORGAN_MASKS_DIR):
        if fn.endswith(".npy"):
            done.add(fn[: -len(".npy")])
    return done


def process_one(volume_id, slice_index):
    ct_path = hf_hub_download(REPO_CT, volume_id_to_ct_rate_path(volume_id), repo_type="dataset")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, f"{volume_id}_organs.nii.gz")
            cmd = [
                "TotalSegmentator",
                "-i", ct_path,
                "-o", out_path,
                "-ml",
                "--fast",
                "-d", "mps",
                "-q",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"TotalSegmentator failed: {result.stderr[-1500:]}")

            seg_data = np.asarray(nib.load(out_path).dataobj)
            ct_shape = nib.load(ct_path).shape
            if seg_data.shape != ct_shape:
                raise ValueError(f"seg shape {seg_data.shape} != ct shape {ct_shape}")

            organ_slice = seg_data[:, :, slice_index].astype(np.uint8)
            np.save(os.path.join(ORGAN_MASKS_DIR, f"{volume_id}.npy"), organ_slice)
    finally:
        cleanup_download(ct_path)


def main():
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    records = load_manifest_records()
    log(f"{len(records)} volumes in manifest.jsonl")

    done = already_done_ids()
    log(f"{len(done)} already have organ masks, will be skipped")

    todo = [r for r in records if r["volume_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"Processing {len(todo)} volumes")

    n_ok, n_fail = 0, 0
    t_start = time.time()
    for i, r in enumerate(todo):
        volume_id, slice_index = r["volume_id"], r["slice_index"]
        t0 = time.time()
        try:
            process_one(volume_id, slice_index)
            n_ok += 1
            log(f"[{i+1}/{len(todo)}] {volume_id} OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            n_fail += 1
            with open(FAILED_PATH, "a") as f:
                f.write(json.dumps({"volume_id": volume_id, "error": str(e)}) + "\n")
            log(f"[{i+1}/{len(todo)}] {volume_id} FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"Done. ok={n_ok} fail={n_fail} elapsed={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
