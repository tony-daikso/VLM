"""
PadChest-GR single-image dataset extractor (X-ray track, independent from the
CT-RATE/HiPaS pipeline under data/CT/).

Turns the raw PadChest-GR release (data/X-ray/PadChest-GR/) into a manifest +
extracted PNGs under data/X-ray/PadChest-GR/processed/, ready for the CXR
Stage 1/2/3 U-VLM track (see U-VLM/cxr/stage*/STAGE*_PLAN.md).

Inputs (already downloaded, see data/X-ray/PadChest-GR/download_all.sh):
- grounded_reports_20240819.json: 4,555 studies, each with a `findings` list
  (sentence_en/es, abnormal, boxes/extra_boxes in 0..1 normalized coords,
  labels, locations, progression). Sentences with abnormal=False often omit
  the boxes/labels/locations keys entirely -- clean_finding() below is
  defensive about that.
- master_table.csv.zip: one row per (study, label) pair -- used here only to
  pull the official patient-safe train/validation/test split and the
  label_group taxonomy (26 classes) per study, NOT for per-row content,
  since its granularity doesn't line up 1:1 with the JSON findings.
- Padchest_GR_files/PadChest_GR.zip.001..037: the actual PNGs, split into 37
  parts purely for download purposes (each part starts with a plain zip
  local-file-header, i.e. simple byte-concatenation, not a zip -s multi-disk
  archive). ConcatFilesReader below presents them as one seekable stream so
  zipfile can read the combined ~36GB archive without first cat'ing it into
  a duplicate file on disk.

PadChest_GR_progression_prior_studies/ (prior-study images for longitudinal
comparison) is intentionally not touched -- out of scope for this pass, see
U-VLM/cxr/stage1/STAGE1_PLAN.md.

Resumable: skips study_ids already present in manifest.jsonl and images
already extracted on disk.
"""
import argparse
import bisect
import csv
import io
import json
import os
import zipfile

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "X-ray", "PadChest-GR")
GROUNDED_JSON_PATH = os.path.join(RAW_DIR, "grounded_reports_20240819.json")
MASTER_TABLE_ZIP = os.path.join(RAW_DIR, "master_table.csv.zip")
ZIP_PARTS_DIR = os.path.join(RAW_DIR, "Padchest_GR_files")

OUT_DIR = os.path.join(RAW_DIR, "processed")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
FAILED_PATH = os.path.join(OUT_DIR, "manifest_failed.jsonl")
LABEL_GROUPS_PATH = os.path.join(OUT_DIR, "label_groups.json")

os.makedirs(IMAGES_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


class ConcatFilesReader:
    """Read a sequence of files, in order, as one seekable binary stream."""

    def __init__(self, paths):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.total_size = self.offsets[-1]
        self._pos = 0
        self._fh = None
        self._fh_idx = -1

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.total_size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def _open_part(self, idx):
        if idx != self._fh_idx:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.paths[idx], "rb")
            self._fh_idx = idx
        return self._fh

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.total_size - self._pos
        out = bytearray()
        remaining = size
        while remaining > 0 and self._pos < self.total_size:
            idx = bisect.bisect_right(self.offsets, self._pos) - 1
            local_off = self._pos - self.offsets[idx]
            fh = self._open_part(idx)
            fh.seek(local_off)
            to_read = min(remaining, self.sizes[idx] - local_off)
            chunk = fh.read(to_read)
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            remaining -= len(chunk)
        return bytes(out)

    def close(self):
        if self._fh is not None:
            self._fh.close()


def open_padchest_zip():
    parts = sorted(p for p in os.listdir(ZIP_PARTS_DIR) if p.startswith("PadChest_GR.zip."))
    if not parts:
        raise FileNotFoundError(f"no PadChest_GR.zip.* parts found in {ZIP_PARTS_DIR}")
    paths = [os.path.join(ZIP_PARTS_DIR, p) for p in parts]
    total_gb = sum(os.path.getsize(p) for p in paths) / 1e9
    log(f"combining {len(paths)} zip parts ({total_gb:.1f} GB) into one virtual stream")
    return zipfile.ZipFile(ConcatFilesReader(paths))


def load_master_table():
    """Returns {study_id: {"split": str, "patient_id": str, "label_groups": set}}."""
    with zipfile.ZipFile(MASTER_TABLE_ZIP) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))

    by_study = {}
    for row in rows:
        sid = row["StudyID"]
        entry = by_study.setdefault(
            sid, {"split": row["split"], "patient_id": row["PatientID"], "label_groups": set()}
        )
        if row["label_group"]:
            entry["label_groups"].add(row["label_group"])
    return by_study


def already_done_ids():
    done = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["study_id"])
    return done


def clean_finding(f):
    return {
        "sentence_en": f.get("sentence_en"),
        "sentence_es": f.get("sentence_es"),
        "abnormal": f.get("abnormal", False),
        "boxes": f.get("boxes", []),
        "extra_boxes": f.get("extra_boxes", []),
        "labels": f.get("labels", []),
        "locations": f.get("locations", []),
        "progression": f.get("progression"),
    }


def process_study(study, master_row, all_label_groups, name_by_basename, zf):
    study_id = study["StudyID"]
    image_id = study["ImageID"]

    zip_entry = name_by_basename.get(image_id)
    if zip_entry is None:
        raise FileNotFoundError(f"{image_id} not found in PadChest_GR zip")

    out_path = os.path.join(IMAGES_DIR, image_id)
    if not os.path.exists(out_path):
        with open(out_path, "wb") as out_f:
            out_f.write(zf.read(zip_entry))
        Image.open(out_path).load()  # integrity check; raises on a corrupt/truncated PNG

    findings = [clean_finding(f) for f in study.get("findings", [])]
    present = master_row["label_groups"]

    return {
        "study_id": study_id,
        "image_id": image_id,
        "image_relpath": os.path.join("images", image_id),
        "split": master_row["split"],
        "patient_id": master_row["patient_id"],
        "findings": findings,
        "classification_labels": {lg: int(lg in present) for lg in all_label_groups},
        "report_text": " ".join(f["sentence_en"] for f in findings if f["sentence_en"]),
        "abnormal_study": any(f["abnormal"] for f in findings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N studies (for testing)")
    args = parser.parse_args()

    log("loading grounded_reports_20240819.json ...")
    with open(GROUNDED_JSON_PATH) as f:
        studies = json.load(f)
    log(f"{len(studies)} studies in JSON")

    log("loading master_table.csv.zip ...")
    master_table = load_master_table()
    log(f"{len(master_table)} studies in master table")

    all_label_groups = sorted({lg for entry in master_table.values() for lg in entry["label_groups"]})
    log(f"{len(all_label_groups)} label_group classes: {all_label_groups}")

    done = already_done_ids()
    log(f"{len(done)} studies already in manifest, will be skipped")

    if args.limit:
        studies = studies[: args.limit]

    manifest_f = open(MANIFEST_PATH, "a")
    failed_f = open(FAILED_PATH, "a")
    n_ok, n_fail = 0, 0
    try:
        with open_padchest_zip() as zf:
            name_by_basename = {os.path.basename(n): n for n in zf.namelist() if n.lower().endswith(".png")}
            log(f"{len(name_by_basename)} PNGs found in archive")

            for i, study in enumerate(studies):
                study_id = study["StudyID"]
                if study_id in done:
                    continue
                master_row = master_table.get(study_id)
                if master_row is None:
                    n_fail += 1
                    failed_f.write(json.dumps({"study_id": study_id, "error": "missing from master_table"}) + "\n")
                    failed_f.flush()
                    continue
                try:
                    record = process_study(study, master_row, all_label_groups, name_by_basename, zf)
                    manifest_f.write(json.dumps(record) + "\n")
                    manifest_f.flush()
                    n_ok += 1
                    if n_ok % 250 == 0:
                        log(f"[{i+1}/{len(studies)}] {n_ok} ok, {n_fail} failed so far")
                except Exception as e:
                    n_fail += 1
                    failed_f.write(json.dumps({"study_id": study_id, "error": str(e)}) + "\n")
                    failed_f.flush()
                    log(f"FAILED {study_id}: {e}")
    finally:
        manifest_f.close()
        failed_f.close()

    with open(LABEL_GROUPS_PATH, "w") as f:
        json.dump(all_label_groups, f, indent=2)

    log(f"Done. ok={n_ok} fail={n_fail}. label_groups written to {LABEL_GROUPS_PATH}")


if __name__ == "__main__":
    main()
