import os
import argparse
import collections
import pydicom
import SimpleITK as sitk


def is_candidate_series(image_type):
    # keep true acquired axial CT slices, drop localizer/topogram/scout,
    # derived reconstructions (MPR) and secondary captures (protocol sheets)
    image_type = set(image_type)
    if "LOCALIZER" not in image_type and "DERIVED" not in image_type and "SECONDARY" not in image_type:
        return True
    return False


def pick_series(series_info):
    # series_info: dict uid -> dict(count, desc, files)
    candidates = {uid: info for uid, info in series_info.items() if info["is_candidate"]}
    if not candidates:
        candidates = series_info

    non_contrast_markers = ["c-", "非增强", "plain"]

    def is_noncontrast(desc):
        d = desc.lower()
        return any(m in d for m in non_contrast_markers)

    contrast_candidates = {uid: info for uid, info in candidates.items() if not is_noncontrast(info["desc"])}
    pool = contrast_candidates if contrast_candidates else candidates

    max_count = max(info["count"] for info in pool.values())
    tied = {uid: info for uid, info in pool.items() if info["count"] == max_count}

    if len(tied) > 1:
        for uid, info in tied.items():
            if "delay" in info["desc"].lower():
                return uid, info
        for uid, info in tied.items():
            if "2nd" in info["desc"].lower():
                return uid, info

    uid, info = next(iter(tied.items()))
    return uid, info


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-patient folders of raw DICOM slices (possibly containing "
                     "multiple series/phases) into a single representative NIfTI file per patient."
    )
    parser.add_argument("--src_dir", required=True, help="Folder containing one subfolder of .dcm files per patient.")
    parser.add_argument("--dst_dir", required=True, help="Output folder for <patient_id>.nii.gz files.")
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    patient_ids = sorted(os.listdir(args.src_dir))
    summary = []

    for pid in patient_ids:
        pdir = os.path.join(args.src_dir, pid)
        series_files = collections.defaultdict(list)
        series_meta = {}

        for f in os.listdir(pdir):
            fp = os.path.join(pdir, f)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
            except Exception:
                continue
            if getattr(ds, "Modality", "") != "CT":
                continue
            uid = ds.SeriesInstanceUID
            series_files[uid].append(fp)
            if uid not in series_meta:
                series_meta[uid] = {
                    "desc": str(getattr(ds, "SeriesDescription", "")),
                    "image_type": list(getattr(ds, "ImageType", [])),
                }

        series_info = {}
        for uid, files in series_files.items():
            meta = series_meta[uid]
            series_info[uid] = {
                "count": len(files),
                "desc": meta["desc"],
                "image_type": meta["image_type"],
                "files": files,
                "is_candidate": is_candidate_series(meta["image_type"]),
            }

        chosen_uid, chosen = pick_series(series_info)

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(pdir, chosen_uid))
        image = reader.Execute()

        out_path = os.path.join(args.dst_dir, f"{pid}.nii.gz")
        sitk.WriteImage(image, out_path)

        summary.append((pid, chosen["desc"], chosen["count"], image.GetSize()))
        print(f"{pid}: chose series '{chosen['desc']}' ({chosen['count']} slices) -> size {image.GetSize()}")

    print("\nDone. Summary:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
