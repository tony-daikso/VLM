"""
RADAR X-ray Phase 6 diagnostic: does the model discriminate at all on its
OWN native training task (in-batch image<->whole_image_caption retrieval,
using the actual multi-label captions it was trained on), independent of
the simplified "normal." vs "<finding>." zero-shot prompt template used by
zero_shot_eval.py?

Motivated by a decoupling observed during the current retrain: loss_itc
keeps decreasing epoch over epoch, but zero-shot AUC (both train-subsample
and val) stays pinned at ~0.49-0.51 (chance level) the whole time. Two
competing explanations:
  1. genuine underfitting / optimization problem -- the model isn't
     learning discriminative image-text alignment at all, on anything.
  2. prompt-template mismatch -- the model DID learn something for its own
     (often multi-label, verbose) training captions, but the zero-shot
     eval's single-label "normal."/"<finding>." prompts are too far out of
     that distribution to probe it.

This script tells them apart by replaying the model's OWN whole-image ITC
forward pass (attention_whole/vision_projs_whole/text_proj, same as
radar_pretrain.py's forward()) on a single big batch of real training
samples with their real whole_image_caption text, then checks: for each
image, does the highest-similarity caption in the batch actually belong to
it (accounting for other samples that happen to share the exact same
caption text, via the same equality-based positive mask used in the loss
fix -- otherwise the ~37% of captions that are literally "normal." would
make retrieval look artificially hard for no real reason). Compared against
the chance-level accuracy implied by each row's own number of positive
(same-caption) candidates, not a flat 1/N, since duplicate captions make
chance recall higher than 1/N to begin with.

Usage:
    cd RADAR/X-ray/phase6
    python3 -u diagnose_retrieval.py [--n 256]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

_PHASE3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3")
_LAVIS_DIR = os.path.join(_PHASE3_DIR, "lavis")
sys.path.insert(0, _PHASE3_DIR)

_fake_lavis = types.ModuleType("lavis")
_fake_lavis.__path__ = [os.path.abspath(_LAVIS_DIR)]
sys.modules["lavis"] = _fake_lavis

from lavis.common.registry import registry  # noqa: E402
registry.register_path("library_root", os.path.abspath(_LAVIS_DIR))

# --- same transformers-compat shim as phase5/train.py and phase6/zero_shot_eval.py ---
import transformers.modeling_utils as _hf_modeling_utils  # noqa: E402
import transformers.pytorch_utils as _hf_pytorch_utils  # noqa: E402

if not hasattr(_hf_modeling_utils, "apply_chunking_to_forward"):
    _hf_modeling_utils.apply_chunking_to_forward = _hf_pytorch_utils.apply_chunking_to_forward
if not hasattr(_hf_modeling_utils, "prune_linear_layer"):
    _hf_modeling_utils.prune_linear_layer = _hf_pytorch_utils.prune_linear_layer
if not hasattr(_hf_modeling_utils, "find_pruneable_heads_and_indices"):
    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
    _hf_modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

_orig_finalize_model_loading = _hf_modeling_utils.PreTrainedModel._finalize_model_loading


@staticmethod
def _patched_finalize_model_loading(model, load_config, loading_info):
    if not hasattr(model, "all_tied_weights_keys"):
        model.post_init()
    return _orig_finalize_model_loading(model, load_config, loading_info)


_hf_modeling_utils.PreTrainedModel._finalize_model_loading = _patched_finalize_model_loading

if not hasattr(_hf_modeling_utils.PreTrainedModel, "get_extended_attention_mask"):
    def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def _invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        else:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        dtype = next(self.parameters()).dtype
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
        return encoder_extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return head_mask.to(dtype=next(self.parameters()).dtype)

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    _hf_modeling_utils.PreTrainedModel.get_extended_attention_mask = _get_extended_attention_mask
    _hf_modeling_utils.PreTrainedModel.invert_attention_mask = _invert_attention_mask
    _hf_modeling_utils.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    _hf_modeling_utils.PreTrainedModel.get_head_mask = _get_head_mask

from lavis.models.med import XBertEncoder  # noqa: E402
from lavis.models.radar_models.radar_pretrain import RadarPretrain  # noqa: E402

sys.path.insert(0, os.path.join(_LAVIS_DIR, "models", "radar_models"))
from vision_branch import VisionBranch  # noqa: E402

_PHASE4_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4")
sys.path.insert(0, _PHASE4_DIR)
from dataset import RadarXrayDataset  # noqa: E402

from torch.utils.data import DataLoader

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt", "checkpoint_radar_pretrain_xray.pth")


def _fix_position_ids_buffer(bert_module):
    # see phase5/train.py's build_model() for the full explanation
    emb = bert_module.embeddings
    n = emb.position_ids.shape[-1]
    emb.position_ids.copy_(torch.arange(n, device=emb.position_ids.device).expand_as(emb.position_ids))


def build_model(device):
    image_encoder = VisionBranch()
    text_encoder = XBertEncoder.from_config({"med_config_path": "unused"}, from_pretrained=True)
    _fix_position_ids_buffer(text_encoder)
    model = RadarPretrain(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        text_decoder=None,
        queue_size=0,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=False,
        max_txt_len=512,
        radar_plus=True,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=True)
    print(f"loaded {CKPT_PATH}: {msg}")
    model.eval()
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256, help="batch size for the single retrieval batch")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    ds = RadarXrayDataset(split=args.split)
    loader = DataLoader(ds, batch_size=args.n, shuffle=True, num_workers=4)
    batch = next(iter(loader))

    image = batch["image"].to(device)
    reports = batch["text_input"]["report"]  # real, native training captions (often multi-label)
    n = len(reports)
    print(f"batch: {n} samples from {args.split} split")
    print(f"  captions unique: {len(set(reports))} / {n} ({100*len(set(reports))/n:.1f}%)")

    _, _, embeds1, embeds2, embeds3, _, _, _ = model.visual_encoder(image, None)
    key_whole = value_whole = torch.cat([embeds1, embeds2, embeds3], dim=1)
    query_whole = model.query_tokens_whole.unsqueeze(0).expand(key_whole.shape[0], -1, -1)
    image_feat_whole, _ = model.attention_whole(query_whole, key_whole, value_whole)
    image_feat_whole = image_feat_whole.squeeze(1)
    image_feat_whole = F.normalize(model.vision_projs_whole(image_feat_whole), dim=-1)

    text = model.tokenizer(list(reports), padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
    text_output = model.text_encoder.forward_text(text)
    text_feat_whole = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

    sim_i2t = (image_feat_whole @ text_feat_whole.t() / model.temp).cpu().numpy()
    sim_t2i = sim_i2t.T

    reports_arr = np.array(reports)
    same_caption = reports_arr[:, None] == reports_arr[None, :]  # includes the diagonal

    def eval_direction(sim, positive_mask, name):
        top1 = sim.argmax(axis=1)
        n_positives_per_row = positive_mask.sum(axis=1)
        correct = positive_mask[np.arange(n), top1]
        acc = correct.mean()
        chance_acc = (n_positives_per_row / n).mean()  # expected accuracy of a RANDOM ranking
        print(f"  {name}: top1_acc={acc:.4f}  chance_baseline={chance_acc:.4f}  "
              f"(avg {n_positives_per_row.mean():.1f} positive candidates/row out of {n})")
        return acc, chance_acc

    print("\n=== native-caption retrieval accuracy (this model's own training task, no zero-shot prompt template) ===")
    i2t_acc, i2t_chance = eval_direction(sim_i2t, same_caption, "image->text")
    t2i_acc, t2i_chance = eval_direction(sim_t2i, same_caption, "text->image")

    avg_lift = ((i2t_acc - i2t_chance) + (t2i_acc - t2i_chance)) / 2
    print(f"\naverage lift over chance: {avg_lift:+.4f}")
    if avg_lift < 0.05:
        print("--> at or near chance: model has NOT learned meaningful image-text alignment even on its own training format (points to genuine underfitting/optimization problem, not just a zero-shot prompt-template mismatch)")
    else:
        print("--> clearly above chance: model HAS learned something on its native training format (points toward prompt-template mismatch as the zero-shot AUC culprit, not pure underfitting)")


if __name__ == "__main__":
    main()
