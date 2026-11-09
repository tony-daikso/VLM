"""Method 4 dataset / collate. DataLoader must use pin_memory=False (A6000 pinned copies corrupt ~6% of batches)."""
from __future__ import annotations

import json
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from m4_common import (BACKBONES, DISEASES, LEVEL_RE_TXT, LR_PAIRS, N_IMG_TOKENS, NUM_UNIT_RE, SYSTEM,
                       drop_measurements, gt_cue_probs, replace_cue, swap_lr)


def lr_permutation(meta_rows):
    """token index -> index of its left/right partner (same region / window / pool); unpaired tokens map to self."""
    idx = {(r["group"], r["unit"], r["region"], r["window"], r["pool"]): int(r["tok"]) for r in meta_rows}
    perm = list(range(len(meta_rows)))
    for g, a, b in LR_PAIRS:
        for (gg, u, reg, w, p), i in idx.items():
            if gg == g and u == a and (g, b, reg, w, p) in idx:
                j = idx[(g, b, reg, w, p)]
                perm[i], perm[j] = j, i
    return torch.tensor(perm)


def attr_char_spans(text, terms):
    """character spans of attribute words (B's values + synonyms), numbers with cm/mm and vertebral levels."""
    spans = []
    pats = [re.escape(t) for t in terms if t] + [NUM_UNIT_RE, LEVEL_RE_TXT]
    for m in re.finditer(r"(?i)(?<![A-Za-z])(?:" + "|".join(pats) + r")(?![A-Za-z])", text):
        spans.append(m.span())
    return spans

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


class CueGT:
    """--cue-source gt_dropout (train rows only): the classifier line is rebuilt from the report labels of the case
    (teacher forcing; m4_common.gt_cue_probs), each finding dropped with probability `dropout`, resampled every time
    the item is read. Labels come from build_cue_gt.py, which holds TRAIN labels only."""

    def __init__(self, path, dropout=0.3, gt_map="sample", hi=0.95, lo=0.02):
        z = json.load(open(path))
        self.labels, self.pools, self.thresholds = z["labels_train"], z["pools"], z["thresholds"]
        self.dropout, self.gt_map, self.hi, self.lo = dropout, gt_map, hi, lo

    def probs(self, case_id, rng):
        if case_id not in self.labels:
            raise SystemExit(f"{case_id}: no train label in the cue file (gt cues are train-only)")
        return gt_cue_probs(self.labels[case_id], self.pools, rng, self.dropout, self.gt_map, self.hi, self.lo)


class M4Dataset(Dataset):
    def __init__(self, jsonl, tokenizer, use_image=True, max_len=1024, n_img=N_IMG_TOKENS, limit=0,
                 feat_root=None, backbone="lingshu7b", allow_test=False, train=True, meta_rows=None,
                 unit_labels=None, lr_flip_prob=0.0, attr_weight=1.0, cue_gt=None, meas_dropout=0.0):
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
        # experiment switches (all off by default -> identical items to the baseline)
        self.lr_flip_prob = lr_flip_prob if train else 0.0
        self.attr_weight = attr_weight if train else 1.0
        self.unit_labels = unit_labels            # {case_id: {"group/unit": [[disease_index, 0/1], ...]}} or None
        self.cue_gt = cue_gt if train else None   # CueGT or None (None = the prompt's own OOF / classifier line)
        self.meas_dropout = meas_dropout if train else 0.0
        if self.cue_gt is not None:
            bad = [r["case_id"] for r in self.rows if r["split"] != "train"]
            if bad:
                raise SystemExit(f"gt cues are train-only; {len(bad)} non-train rows (e.g. {bad[0]})")
        if meta_rows is not None:
            self.T = len(meta_rows)
            self.perm = lr_permutation(meta_rows)
            self.unit_tok = {}
            for r in meta_rows:
                self.unit_tok.setdefault(f"{r['group']}/{r['unit']}", []).append(int(r["tok"]))
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

    def unit_label_tensor(self, cid):
        y = torch.full((self.T, len(DISEASES)), -1, dtype=torch.int8)
        for key, pairs in (self.unit_labels.get(cid) or {}).get("units", {}).items():
            for t in self.unit_tok.get(key, []):
                for d, v in pairs:
                    y[t, d] = v
        return y

    def __getitem__(self, i):
        r = self.rows[i]
        flip = self.lr_flip_prob > 0 and random.random() < self.lr_flip_prob   # torch seeds `random` per worker
        prompt, target = r["prompt"], r.get("target", "")
        cue_info = None
        if self.cue_gt is not None or self.meas_dropout > 0:
            cue_info = {}
            if self.cue_gt is not None:
                probs, n_lab, n_drop = self.cue_gt.probs(r["case_id"], random)
                prompt = replace_cue(prompt, probs)
                cue_info.update(probs=probs, n_labelled=n_lab, n_dropped=n_drop)
            if self.meas_dropout > 0:
                prompt, n_m, k_m = drop_measurements(prompt, self.meas_dropout, random)
                cue_info.update(n_meas=n_m, n_meas_dropped=k_m)
        if flip:
            prompt, target = swap_lr(prompt), swap_lr(target)
        p = self.tk(chatml(prompt, self.use_image, self.n_img, self.backbone), add_special_tokens=False)["input_ids"]
        item = {"case_id": r["case_id"], "prompt_ids": p}
        if self.train:
            if self.attr_weight != 1.0:
                terms = [swap_lr(x) for x in r.get("attr_terms", [])] if flip else r.get("attr_terms", [])
                enc = self.tk(target, add_special_tokens=False, return_offsets_mapping=True)
                spans = attr_char_spans(target, terms)
                tw = [self.attr_weight if any(a < e and s_ < b for s_, e in spans) else 1.0
                      for a, b in enc["offset_mapping"]]
                t = enc["input_ids"] + self.end_ids
                tw = tw + [1.0] * len(self.end_ids)
            else:
                t = self.tk(target, add_special_tokens=False)["input_ids"] + self.end_ids
                tw = None
            n = max(self.max_len - len(p), 1)
            t = t[:n]                                            # never cut the prompt (it holds the image tokens)
            item["input_ids"], item["labels"] = p + t, [-100] * len(p) + t
            if tw is not None:
                item["loss_w"] = [0.0] * len(p) + tw[:n]
        if self.use_image:
            tok, mask = self.features(r)
            if flip:
                assert hasattr(self, "perm"), "lr flip with image tokens needs meta_rows"
                tok, mask = tok[self.perm], mask[self.perm]
            item["tok"], item["mask"] = tok, mask
            assert sum(x == self.img_id for x in p) == self.n_img
            if self.unit_labels is not None and self.train:
                y = self.unit_label_tensor(r["case_id"])
                item["unit_labels"] = y[self.perm] if flip else y
        item["flipped"] = flip
        if cue_info is not None:
            item["cue_info"] = cue_info
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
    if train and "loss_w" in batch[0]:
        w = torch.zeros((len(batch), L), dtype=torch.float32)
        for i, b in enumerate(batch):
            w[i, : len(b["loss_w"])] = torch.tensor(b["loss_w"])
        out["loss_w"] = w
    if "tok" in batch[0]:
        out["tok"] = torch.stack([b["tok"] for b in batch])
        out["mask"] = torch.stack([b["mask"] for b in batch])
    if "unit_labels" in batch[0]:
        out["unit_labels"] = torch.stack([b["unit_labels"] for b in batch])
    return out
