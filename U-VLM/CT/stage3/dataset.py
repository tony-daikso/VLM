"""
Stage 3 Dataset：讀 processed_rex（CT-RATE）manifest.jsonl，輸出 image + tokenized report。

Image pipeline（windowing/resize/augment）跟 Stage 1/2 完全共用同一套邏輯，不重寫（見
STAGE3_PLAN.md 第 3 節）。跟 Stage 2 的差異只在 target 從 18 維 label 換成文字 token 序列。

Target text 格式固定為 "Findings: {...}\\nImpressions: {...}"（跟 Stage 2 dataset.py 的
format_report_text 用同一組欄位：Findings_EN + Impressions_EN，不含 Clinical/Technique 兩段
制式描述）。Tokenizer 借用現成的 GPT-2 BPE 詞表（只借詞表，decoder 權重從零訓練，見
STAGE3_PLAN.md 第 2.4 節）。report 序列排列為 [BOS] + report tokens + [EOS]，
截斷/補齊到 config["text"]["max_report_tokens"]。
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_report_text(report_text):
    if not isinstance(report_text, dict):
        text = str(report_text or "")
    else:
        findings = report_text.get("Findings_EN", "")
        impressions = report_text.get("Impressions_EN", "")
        text = f"Findings: {findings}\nImpressions: {impressions}"
    return text


def window_normalize(image_hu, window_level, window_width):
    """HU 值 -> 肺窗 windowing -> normalize 到 [-1, 1]。"""
    lo = window_level - window_width / 2
    hi = window_level + window_width / 2
    image = np.clip(image_hu, lo, hi)
    image = (image - lo) / (hi - lo)
    image = image * 2 - 1
    return image.astype(np.float32)


def build_tokenizer(tokenizer_name):
    tok = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class Stage3Dataset(Dataset):
    def __init__(self, config, split, augment, tokenizer=None):
        self.config = config
        self.split = split
        self.augment = augment
        self.resolution = config["image"]["resolution"]
        self.window_level = config["image"]["window_level"]
        self.window_width = config["image"]["window_width"]

        text_cfg = config["text"]
        self.max_report_tokens = text_cfg["max_report_tokens"]
        self.tokenizer = tokenizer or build_tokenizer(text_cfg["tokenizer_name"])
        self.bos_id = self.tokenizer.eos_token_id  # GPT-2 沒有獨立 BOS，沿用 eos id 當 BOS
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        instruction_ids = self.tokenizer(text_cfg["instruction"])["input_ids"]
        self.instruction_ids = torch.tensor(instruction_ids, dtype=torch.long)

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rex_dir = os.path.join(root_dir, config["data"]["processed_rex_dir"])

        splits_path = os.path.join(root_dir, config["data"]["splits_path"])
        with open(splits_path) as f:
            splits = json.load(f)
        ctrate_ids = set(splits["ctrate"][split])

        records = load_jsonl(os.path.join(self.rex_dir, "manifest.jsonl"))
        self.records = sorted(
            (r for r in records if r["volume_id"] in ctrate_ids), key=lambda r: r["volume_id"]
        )

    def __len__(self):
        return len(self.records)

    def _tokenize_report(self, text):
        body_ids = self.tokenizer(text, truncation=True, max_length=self.max_report_tokens - 2)["input_ids"]
        ids = [self.bos_id] + body_ids + [self.eos_id]
        mask = [1] * len(ids)
        pad_len = self.max_report_tokens - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            mask = mask + [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = np.load(os.path.join(self.rex_dir, r["image_npy"]))
        image = window_normalize(image, self.window_level, self.window_width)

        params = self._sample_augment_params() if self.augment else None
        image = self._resize(image[None], mode="bilinear", params=params)[0]

        text = format_report_text(r.get("report_text"))
        report_ids, report_mask = self._tokenize_report(text)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "instruction_ids": self.instruction_ids,
            "report_ids": report_ids,
            "report_mask": report_mask,
            "item_id": r["volume_id"],
            "report_text": text,
        }

    def _sample_augment_params(self):
        aug_cfg = self.config["augmentation"]
        return {
            "hflip": random.random() < aug_cfg["hflip_prob"],
            "angle": random.uniform(-aug_cfg["rotation_deg"], aug_cfg["rotation_deg"]),
            "scale": 1.0 + random.uniform(-aug_cfg["scale_jitter"], aug_cfg["scale_jitter"]),
            "intensity_shift": random.uniform(-aug_cfg["intensity_jitter"], aug_cfg["intensity_jitter"]),
        }

    def _resize(self, array, mode, params):
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode=mode, align_corners=False)

        if params is not None:
            if params["hflip"]:
                tensor = torch.flip(tensor, dims=[-1])

            theta = self._affine_theta(params["angle"], params["scale"])
            grid = F.affine_grid(theta, tensor.shape, align_corners=False)
            tensor = F.grid_sample(tensor, grid, mode=mode, align_corners=False, padding_mode="zeros")
            tensor = tensor * (1 + params["intensity_shift"])

        return tensor[0].numpy()

    @staticmethod
    def _affine_theta(angle_deg, scale):
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor(
            [[cos_a / scale, -sin_a / scale, 0.0], [sin_a / scale, cos_a / scale, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        return theta
