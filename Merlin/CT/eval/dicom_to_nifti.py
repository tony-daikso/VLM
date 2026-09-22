"""Convert the 72 abdomen CT DICOM series to NIfTI for Merlin's DataLoader."""
import os
import SimpleITK as sitk

SRC_ROOT = "/datadrive/VLM/data/CT/CG/abdomen_extracted/abdomen/image"
OUT_DIR = "/datadrive/VLM/Merlin/CT/nifti"

os.makedirs(OUT_DIR, exist_ok=True)

patient_ids = sorted(os.listdir(SRC_ROOT), key=lambda x: int(x))
print(f"Found {len(patient_ids)} patients")

reader = sitk.ImageSeriesReader()
failed = []
for pid in patient_ids:
    src_dir = os.path.join(SRC_ROOT, pid)
    out_path = os.path.join(OUT_DIR, f"{pid}.nii.gz")
    if os.path.exists(out_path):
        continue
    try:
        series_ids = reader.GetGDCMSeriesIDs(src_dir)
        if not series_ids:
            raise RuntimeError("no series found")
        file_names = reader.GetGDCMSeriesFileNames(src_dir, series_ids[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        sitk.WriteImage(image, out_path)
        print(f"patient {pid}: {image.GetSize()} -> {out_path}")
    except Exception as e:
        print(f"patient {pid}: FAILED - {e}")
        failed.append(pid)

print(f"\nDone. Failed: {len(failed)} {failed}")
