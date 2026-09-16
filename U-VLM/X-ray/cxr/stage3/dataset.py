"""
Stage 3 Dataset：讀 data/X-ray/PadChest-GR(-small)/processed/manifest.jsonl，輸出 image +
tokenized report_text（見 STAGE3_PLAN.md 第 1、2 節）。

image 的載入/normalize/cache 邏輯跟 Stage 1/2/2.5 完全一致（同一個 cache 目錄）。沒有
augmentation -- v1 encoder 凍結，report generation 這邊不需要影像增強（見 STAGE3_PLAN.md
第 6 節）。

每筆樣本輸出：
  image: (1, R, R) float32，[-1, 1]
  input_ids / attention_mask: (max_length,) long，GPT2Tokenizer 編碼的 report_text
    （開頭是 eos_token 當 BOS，結尾補一個 eos_token 當停止訊號，padding 用 eos_token_id 補齊，
    dataset 這裡不做 label masking，train.py 的 collate 會把 padding 位置的 label 設成 -100）
  report_text: 原始字串，給 eval.py 當 reference
  study_id: str
  classification_labels: dict，給 eval.py 的 keyword 臨床 proxy 指標用
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile_normalize(arr, lo_pct, hi_pct):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 2 - 1).astype(np.float32)


class Stage3Dataset(Dataset):
    def __init__(self, config, split, tokenizer):
        self.config = config
        self.resolution = config["image"]["resolution"]
        self.pct_lo = config["image"]["percentile_low"]
        self.pct_hi = config["image"]["percentile_high"]
        self.max_length = config["text"]["max_length"]
        self.tokenizer = tokenizer

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        self.data_dir = os.path.join(root_dir, config["data"]["processed_dir"])

        self.cache_dir = os.path.join(
            self.data_dir, f"_cache_r{self.resolution}_p{self.pct_lo}_{self.pct_hi}"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.records = load_jsonl(os.path.join(self.data_dir, "manifest.jsonl"))
        self.records = [r for r in self.records if r["split"] == split]
        if not self.records:
            raise ValueError(f"no rows with split={split!r} in {self.data_dir}/manifest.jsonl")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image = self._load_image(row)

        report_text = row.get("report_text") or ""
        encoded = self.tokenizer(
            report_text,
            max_length=self.max_length - 2,  # 留兩格給開頭的 BOS 跟結尾的 EOS
            truncation=True,
        )
        # 結尾補 EOS，讓模型學會在生成完報告後主動停止（GPT2 的 bos/eos 是同一個 token，
        # 靠位置分辨：開頭那個當 BOS，結尾那個當停止訊號）
        input_ids = [self.tokenizer.bos_token_id] + encoded["input_ids"] + [self.tokenizer.eos_token_id]

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "input_ids": input_ids,  # 變長 list，collate_fn 統一 padding
            "report_text": report_text,
            "study_id": row["study_id"],
            "classification_labels": row.get("classification_labels", {}),
        }

    def _load_image(self, row):
        cache_path = os.path.join(self.cache_dir, f"{row['study_id']}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        img_path = os.path.join(self.data_dir, row["image_relpath"])
        arr = np.array(Image.open(img_path))
        arr = percentile_normalize(arr, self.pct_lo, self.pct_hi)
        tensor = torch.from_numpy(arr)[None, None]
        tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        resized = tensor[0, 0].numpy()

        tmp_path = cache_path + f".tmp{os.getpid()}"
        with open(tmp_path, "wb") as f:
            np.save(f, resized)
        os.replace(tmp_path, cache_path)
        return resized


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            ids = b["input_ids"]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(ids, dtype=torch.long)

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report_text": [b["report_text"] for b in batch],
            "study_id": [b["study_id"] for b in batch],
            "classification_labels": [b["classification_labels"] for b in batch],
        }

    return collate_fn
