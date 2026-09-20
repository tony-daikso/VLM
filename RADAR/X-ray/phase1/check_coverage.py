"""
Join our PA/AP PNG filename list against CheXmask's Padchest.csv to answer:
- what fraction of our 96,270 PA/AP images have a CheXmask mask
- how the Dice RCA (Mean) quality score is distributed on our subset
- whether CSV Height/Width actually matches the real PNG dimensions
- which of our filenames are NOT covered (written to missing_from_chexmask.txt,
  for a later decision on whether the ianpan/chest-x-ray-basic fallback is
  worth building)

Reads Padchest.csv as a stream (csv.DictReader) rather than pandas.read_csv,
since the file also carries a giant Landmarks/RLE payload per row that we
don't want to materialize in memory for a coverage check.
"""
import csv
import io
import os
import sys
import tarfile

from PIL import Image

ROOT_DIR = "/datadrive/VLM/data/X-ray/PadChest-Origin"
TAR_PATH = os.path.join(ROOT_DIR, "PNG_tar", "PADCHEST_PA_AP.tar")
IDS_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "padchest_pa_ap_ids.txt")
# override lets us dry-run against a partial download (e.g. parts/part_00, a
# valid from-byte-0 prefix) before the full CSV finishes fetching
CSV_PATH = os.environ.get("CHEXMASK_CSV_PATH", os.path.join(ROOT_DIR, "chexmask_raw", "Padchest.csv"))
OUT_DIR = os.path.join(ROOT_DIR, "processed_chexmask")
MISSING_PATH = os.path.join(OUT_DIR, "missing_from_chexmask.txt")

# raise the csv module's field-size cap: the Landmarks column can be huge
csv.field_size_limit(sys.maxsize)


def load_our_ids():
    with open(IDS_PATH) as f:
        return set(line.strip() for line in f if line.strip())


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def spot_check_dims(matched_rows, tf, n=15):
    print(f"\nspot-checking Height/Width vs actual PNG size for {n} images...")
    names_in_tar = {os.path.basename(m.name): m for m in tf.getmembers()}
    checked, mismatches = 0, 0
    for image_id, (height, width) in list(matched_rows.items())[:n]:
        member = names_in_tar.get(image_id)
        if member is None:
            continue
        with tf.extractfile(member) as fh:
            img = Image.open(io.BytesIO(fh.read()))
            actual_w, actual_h = img.size
        checked += 1
        ok = (actual_h == height and actual_w == width)
        if not ok:
            mismatches += 1
        print(f"  {image_id}: csv=({height},{width}) actual=({actual_h},{actual_w}) {'OK' if ok else 'MISMATCH'}")
    print(f"spot check: {checked} checked, {mismatches} mismatches")


def main():
    our_ids = load_our_ids()
    print(f"{len(our_ids)} PA/AP images in our tar")

    matched = {}
    rca_means = []
    rca_maxes = []
    n_rows = 0

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if n_rows % 20000 == 0:
                print(f"  ...scanned {n_rows} CheXmask rows")
            if row.get("Height") is None or row.get("ImageID") is None:
                # truncated final row -- only expected when CSV_PATH is a
                # still-downloading partial file (dry-run testing)
                print(f"  skipping malformed row at n_rows={n_rows} (likely truncated EOF)")
                continue
            image_id = row["ImageID"]
            if image_id not in our_ids:
                continue
            rca_mean = float(row["Dice RCA (Mean)"])
            rca_max = float(row["Dice RCA (Max)"])
            matched[image_id] = (int(row["Height"]), int(row["Width"]))
            rca_means.append(rca_mean)
            rca_maxes.append(rca_max)

    print(f"\n{n_rows} total rows in CheXmask Padchest.csv")
    print(f"{len(matched)} / {len(our_ids)} of our PA/AP images matched ({100 * len(matched) / len(our_ids):.2f}%)")

    missing = sorted(our_ids - matched.keys())
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MISSING_PATH, "w") as f:
        f.write("\n".join(missing) + ("\n" if missing else ""))
    print(f"{len(missing)} not covered by CheXmask, written to {MISSING_PATH}")

    rca_means.sort()
    below_07 = sum(1 for v in rca_means if v <= 0.7)
    print("\nDice RCA (Mean) distribution on matched images:")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{p}: {percentile(rca_means, p):.4f}")
    print(f"  <= 0.7 threshold: {below_07} ({100 * below_07 / len(rca_means):.2f}%)")

    with tarfile.open(TAR_PATH) as tf:
        spot_check_dims(matched, tf)


if __name__ == "__main__":
    main()
