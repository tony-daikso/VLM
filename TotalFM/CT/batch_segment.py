"""Run TotalSegmentator (multilabel) on all 72 abdomen CT NIfTI volumes."""
import os
from totalsegmentator.python_api import totalsegmentator

NIFTI_DIR = "/datadrive/VLM/Merlin/CT/nifti"
OUT_DIR = "/datadrive/VLM/TotalFM/CT/segmentations"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    patient_ids = sorted(
        [f[:-len(".nii.gz")] for f in os.listdir(NIFTI_DIR) if f.endswith(".nii.gz")],
        key=lambda x: int(x),
    )
    print(f"{len(patient_ids)} patients to segment")

    for i, pid in enumerate(patient_ids):
        out_path = os.path.join(OUT_DIR, f"{pid}.nii.gz")
        if os.path.exists(out_path):
            print(f"[{i + 1}/{len(patient_ids)}] patient {pid}: already done, skip")
            continue
        ct_path = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
        try:
            totalsegmentator(
                input=ct_path,
                output=out_path,
                task="total",
                fast=True,
                device="gpu",
                ml=True,
                quiet=True,
            )
            print(f"[{i + 1}/{len(patient_ids)}] patient {pid}: done")
        except Exception as e:
            print(f"[{i + 1}/{len(patient_ids)}] patient {pid}: FAILED - {e}")

    print("All done")


if __name__ == "__main__":
    main()
