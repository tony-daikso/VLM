"""Method 4 dataset / collate. DataLoader must use pin_memory=False (A6000 pinned copies corrupt ~6% of batches)."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from m4_common import BACKBONES, N_IMG_TOKENS, SYSTEM

ASSIST_PREFIX = {"chatml": "<|im_start|>assistant\n",
                 "chatml_qwen3_nothink": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}


def chatml(prompt, use_image, n_img=N_IMG_TOKENS, backbone="lingshu7b"):
    """ChatML written out by hand (Lingshu-7B, Lingshu-I-8B and Qwen all use ChatML), so training and inference
    match exactly; only the image block strings differ per backbone."""
    bb = BACKBONES[backbone]
    img = (bb["img_start"] + bb["img_token"] * n_img + bb["img_end"]) if use_image else ""
    user = prompt.replace("<image>", img)
    if not use_image:
        user = "\n".join(l for l in user.splitlines() if l.strip())
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            + ASSIST_PREFIX[bb["chat"]])


class M4Dataset(Dataset):
    def __init__(self, jsonl, tokenizer, use_image=True, max_len=1024, n_img=N_IMG_TOKENS, limit=0,
                 feat_root=None, backbone="lingshu7b", allow_test=False, train=True):
        self.rows = []
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == "test" and not allow_test:
                    raise SystemExit("test split is locked")
                self.rows.append(r)
                if limit and len(self.rows) >= limit:
                    break
        self.tk, self.use_image, self.max_len, self.n_img = tokenizer, use_image, max_len, n_img
        self.feat_root, self.backbone, self.train = feat_root, backbone, train
        self.end_ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
        if use_image:
            tokname = BACKBONES[backbone]["img_token"]
            ids = tokenizer(tokname, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, f"{tokname} must be a single token (add it as a special token)"
            self.img_id = ids[0]

    def __len__(self):
        return len(self.rows)

    def features(self, r):
        p = r["feature_path"]
        if self.feat_root:
            p = self.feat_root.rstrip("/") + "/" + p.split("/features/")[-1]
        z = np.load(p)
        return torch.from_numpy(z["tok"].astype(np.float16)), torch.from_numpy(z["mask"].astype(bool))

    def __getitem__(self, i):
        r = self.rows[i]
        p = self.tk(chatml(r["prompt"], self.use_image, self.n_img, self.backbone), add_special_tokens=False)["input_ids"]
        item = {"case_id": r["case_id"], "prompt_ids": p}
        if self.train:
            t = self.tk(r["target"], add_special_tokens=False)["input_ids"] + self.end_ids
            t = t[: max(self.max_len - len(p), 1)]               # never cut the prompt (it holds the image tokens)
            item["input_ids"], item["labels"] = p + t, [-100] * len(p) + t
        if self.use_image:
            item["tok"], item["mask"] = self.features(r)
            assert sum(x == self.img_id for x in p) == self.n_img
        return item


def collate(batch, pad_id, train=True):
    """train: right padding (input_ids, labels); generation: left padding (prompt only)."""
    key = "input_ids" if train else "prompt_ids"
    L = max(len(b[key]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long) if train else None
    for i, b in enumerate(batch):
        n = len(b[key])
        sl = slice(0, n) if train else slice(L - n, L)
        ids[i, sl] = torch.tensor(b[key])
        att[i, sl] = 1
        if train:
            lab[i, sl] = torch.tensor(b["labels"])
    out = {"case_id": [b["case_id"] for b in batch], "input_ids": ids, "attention_mask": att}
    if train:
        out["labels"] = lab
    if "tok" in batch[0]:
        out["tok"] = torch.stack([b["tok"] for b in batch])
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out
