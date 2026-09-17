"""
把 PadChest-GR manifest.jsonl 轉成標準 LLaVA conversation JSON 格式，圖片欄位指向 Phase 3
產生的 DINO 馬賽克假圖（不是原始 X-ray），跟學長 inference.py 的固定 prompt
"generate analysis report" 保持一致。

輸出：train/validation/test 各一份 JSON，放在 --output_dir 下，給
llava/train/train.py 的 --data_path 用。

用法：
    python3 build_llava_data.py \
        --manifest /root/Desktop/VLM/data/X-ray/PadChest-GR/processed/manifest.jsonl \
        --mosaic_dir /datadrive/VLM/DINO_LLM/X-ray/mosaics \
        --output_dir /datadrive/VLM/DINO_LLM/X-ray/llava_data
"""
import argparse
import json
import os

PROMPT = "generate analysis report"


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_split(rows, split, mosaic_dir):
    out = []
    missing = 0
    for row in rows:
        if row["split"] != split:
            continue
        mosaic_path = os.path.join(mosaic_dir, split, f"{row['study_id']}.png")
        if not os.path.exists(mosaic_path):
            missing += 1
            continue
        out.append({
            "id": row["study_id"],
            "image": mosaic_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{PROMPT}"},
                {"from": "gpt", "value": row.get("report_text") or ""},
            ],
        })
    if missing:
        print(f"[{split}] warning: {missing} studies skipped (mosaic PNG missing)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mosaic_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_manifest(args.manifest)

    for split in ["train", "validation", "test"]:
        data = build_split(rows, split, args.mosaic_dir)
        out_path = os.path.join(args.output_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"{split}: {len(data)} samples -> {out_path}")


if __name__ == "__main__":
    main()
