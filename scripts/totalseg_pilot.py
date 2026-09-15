"""
Pilot test: run TotalSegmentator on ONE CT-RATE volume already in our
manifest, extract the organ mask at the same slice_index we already picked
for that volume, and report timing. This is just to validate the approach
(and measure per-volume cost) before committing to a full-batch run over
all ~400 manifest volumes.

Usage: python scripts/totalseg_pilot.py [volume_id]
If volume_id is omitted, uses the first entry in manifest.jsonl.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download

REPO_CT = "ibrahimhamamci/CT-RATE"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "processed_rex", "manifest.jsonl")


def volume_id_to_ct_rate_path(volume_id):
    parts = volume_id.split("_")
    pid = f"{parts[0]}_{parts[1]}"
    scan = parts[2]
    return f"dataset/valid_fixed/{pid}/{pid}_{scan}/{volume_id}.nii.gz"


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else None
    record = None
    with open(MANIFEST_PATH) as f:
        for line in f:
            d = json.loads(line)
            if target_id is None or d["volume_id"] == target_id:
                record = d
                break
    if record is None:
        print(f"volume_id {target_id} not found in manifest")
        return
    volume_id = record["volume_id"]
    slice_index = record["slice_index"]
    print(f"Pilot volume: {volume_id}, slice_index={slice_index}")

    t0 = time.time()
    ct_path = hf_hub_download(REPO_CT, volume_id_to_ct_rate_path(volume_id), repo_type="dataset")
    t1 = time.time()
    print(f"Download took {t1 - t0:.1f}s -> {ct_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, f"{volume_id}_organs.nii.gz")
        cmd = [
            "TotalSegmentator",
            "-i", ct_path,
            "-o", out_path,
            "-ml",
            "--fast",
            "-d", "mps",
        ]
        print("Running:", " ".join(cmd))
        t2 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        t3 = time.time()
        print(f"TotalSegmentator took {t3 - t2:.1f}s, returncode={result.returncode}")
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-3000:])
            print("STDERR:", result.stderr[-3000:])
            return

        seg_img = nib.load(out_path)
        seg_data = np.asarray(seg_img.dataobj)
        print("Organ seg volume shape:", seg_data.shape)

        ct_img = nib.load(ct_path)
        ct_shape = ct_img.shape
        print("CT volume shape:", ct_shape)

        if seg_data.shape != ct_shape:
            print("WARNING: seg shape does not match CT shape -- check resampling/orientation")

        z = slice_index
        organ_slice = seg_data[:, :, z]
        present_labels = sorted(int(v) for v in np.unique(organ_slice) if v != 0)
        print(f"Slice {z}: {len(present_labels)} distinct organ labels present: {present_labels}")

        pilot_out_dir = os.path.join(ROOT_DIR, "data", "processed_rex", "totalseg_pilot")
        os.makedirs(pilot_out_dir, exist_ok=True)
        np.save(os.path.join(pilot_out_dir, f"{volume_id}_organ_slice.npy"), organ_slice.astype(np.uint8))
        print(f"Saved organ slice to {pilot_out_dir}/{volume_id}_organ_slice.npy")

    try:
        os.remove(ct_path)
    except OSError:
        pass

    print(f"Total pilot time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
