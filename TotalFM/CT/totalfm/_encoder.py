"""Internal nn.Module combining 3D ViT and ModernBERT for inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from totalfm.modernbert import load_modernbert_model
from totalfm.vit_3d import ViT

# Model architecture constants from exp056 training configuration.
_VIT_IMAGE_SIZE = 192
_VIT_PATCH_SIZE = 16
_VIT_FRAMES = 32
_VIT_FRAME_PATCH = 4
_VIT_DIM = 1008
_VIT_DEPTH = 12
_VIT_HEADS = 12
_VIT_MLP_DIM = 3072
_VIT_CHANNELS = 3
_VIT_OUT_DIM = 768
_EMBED_DIM = 768
_TEXT_MODEL = "Alibaba-NLP/gte-modernbert-base"
_MAX_TOKEN_LENGTH = 256


class FoundationModel(nn.Module):
    """Combined image + text encoder for TotalFM inference.

    This module is not intended for direct use; use :class:`totalfm.TotalFM`
    instead.
    """

    def __init__(self) -> None:
        super().__init__()

        self.vit = ViT(
            image_size=_VIT_IMAGE_SIZE,
            image_patch_size=_VIT_PATCH_SIZE,
            frames=_VIT_FRAMES,
            frame_patch_size=_VIT_FRAME_PATCH,
            dim=_VIT_DIM,
            depth=_VIT_DEPTH,
            heads=_VIT_HEADS,
            mlp_dim=_VIT_MLP_DIM,
            channels=_VIT_CHANNELS,
            dim_head=_VIT_DIM // _VIT_HEADS,  # 84
            dropout=0.1,
            emb_dropout=0.1,
            out_dim=_VIT_OUT_DIM,
            reduction="attn_pool",
        )

        text_config, tokenizer, text_encoder = load_modernbert_model(_TEXT_MODEL)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.image_proj = nn.Linear(_VIT_OUT_DIM, _EMBED_DIM)
        self.text_proj = nn.Linear(text_config.hidden_size, _EMBED_DIM)

        # Kept for checkpoint compatibility (not used during inference).
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([1]) * -10)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of CT patches to normalised embeddings.

        Args:
            images: Tensor of shape ``(B, C, H, W, D)``.

        Returns:
            L2-normalised embedding tensor of shape ``(B, 768)`` on CPU.
        """
        device = next(self.parameters()).device
        images = images.to(device)
        features = self.vit(images, None)
        features = self.image_proj(features)
        return F.normalize(features, p=2, dim=-1, eps=1e-6).cpu()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings to normalised embeddings.

        Args:
            texts: List of input strings.

        Returns:
            L2-normalised embedding tensor of shape ``(N, 768)`` on CPU.
        """
        device = next(self.parameters()).device

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_TOKEN_LENGTH,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        outputs = self.text_encoder(**tokenized)
        last = outputs.last_hidden_state  # (B, T, H)
        mask = tokenized["attention_mask"]  # (B, T)

        # Mean pooling, excluding CLS and SEP tokens.
        for special_id in {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id}:
            if special_id is not None:
                mask = mask * (tokenized["input_ids"] != special_id).long()
        mask = mask.unsqueeze(-1)
        features = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        features = features.to(self.text_proj.weight.dtype)
        features = self.text_proj(features)
        return F.normalize(features, p=2, dim=-1, eps=1e-6).cpu()
