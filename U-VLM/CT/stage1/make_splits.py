"""
產生 U-VLM/stage1/splits.json：CT-RATE 跟 HiPaS 各自獨立做 85/15 train/val split。

CT-RATE 用 patient-level split（volume_id 前兩段組成 patient id，例如
"train_409_a_2" -> "train_409"，同一病人的多個 scan/recon 不能跨 train/val）。
HiPaS 用 case_id 直接切（每個 case 只有一張切片，沒有多 scan 問題）。

跑法：python3 U-VLM/stage1/make_splits.py
"""
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

REX_MANIFEST = os.path.join(ROOT_DIR, "data", "processed_rex", "manifest.jsonl")
HIPAS_MANIFEST = os.path.join(ROOT_DIR, "data", "processed_hipas", "manifest.jsonl")
OUT_PATH = os.path.join(SCRIPT_DIR, "splits.json")

SEED = 42
VAL_RATIO = 0.15


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def volume_id_to_patient_id(volume_id):
    parts = volume_id.split("_")
    return f"{parts[0]}_{parts[1]}"


def split_ids(ids, val_ratio, seed):
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def main():
    rex_records = load_jsonl(REX_MANIFEST)
    hipas_records = load_jsonl(HIPAS_MANIFEST)

    patient_to_volumes = {}
    for r in rex_records:
        pid = volume_id_to_patient_id(r["volume_id"])
        patient_to_volumes.setdefault(pid, []).append(r["volume_id"])

    train_patients, val_patients = split_ids(list(patient_to_volumes.keys()), VAL_RATIO, SEED)
    ctrate_train = sorted(v for p in train_patients for v in patient_to_volumes[p])
    ctrate_val = sorted(v for p in val_patients for v in patient_to_volumes[p])

    hipas_ids = [r["case_id"] for r in hipas_records]
    hipas_train, hipas_val = split_ids(hipas_ids, VAL_RATIO, SEED)

    splits = {
        "ctrate": {"train": ctrate_train, "val": ctrate_val},
        "hipas": {"train": hipas_train, "val": hipas_val},
    }

    with open(OUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"CT-RATE: {len(patient_to_volumes)} patients, {len(rex_records)} volumes "
          f"-> train {len(ctrate_train)} / val {len(ctrate_val)} volumes")
    print(f"HiPaS: {len(hipas_records)} cases -> train {len(hipas_train)} / val {len(hipas_val)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
