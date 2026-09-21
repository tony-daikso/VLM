"""
RADAR X-ray Phase 6: zero-shot classification AUC, the last unchecked item
in PLAN.md's Phase 5 ("用 Labels 欄位做 zero-shot 分類評估驗證效果").

Mirrors RADAR/CT/RADAR_train/calc_metrics.py's pattern (built from
infer_merlin_whole.py + infer_merlin_anatomy.py's ZeroShotResult CSVs), but
collapsed into one script since the X-ray val set (4,747 images) is small
enough to just compute everything in one pass instead of writing
intermediate CSVs read back by a second script:
  1. for each finding, embed a positive prompt ("<finding>.") and a negative
     prompt ("normal.") through the trained text encoder -- these are not
     hand-written disease descriptions (unlike CT's DiseasePrompts regex
     sets, which existed to MINE ground truth out of free-text reports).
     PadChest already gives structured ground truth (labels_raw, from
     phase2's manifest), so the prompts only need to match the caption
     STYLE the model was actually trained on: phase2/build_region_captions.py
     builds captions by literally concatenating "<label>." tokens, and a
     clean image's caption is exactly "normal." -- so using that same
     vocabulary as the zero-shot prompt is the closest thing to in-distribution
     text this model has seen, not a stylistic choice.
  2. embed every val image through the trained visual encoder, both as a
     whole image (attention_whole / query_tokens_whole -- same as CT's
     "_whole" CSV) and per-organ (model.forward_test_win, CT's "_anatomy"
     CSV logic, vendored unchanged from CT's radar_pretrain.py) using the
     model's OWN predicted segmentation to decide which organs are
     intact/visible in-frame -- forward_test_win ignores whatever `masks`
     argument is passed in and overwrites it with its own seg_probs, so
     ground-truth CheXmask masks are not needed at eval time.
  3. cosine similarity -> softmax over [negative, positive] -> AUC per
     finding via sklearn, following calc_metrics.py's np.isnan(prob)->0.0
     convention for organs that aren't intact/visible in a given image
     (this is not equivalent to dropping the image -- it counts as a
     confident "no" prediction, so it can hurt AUC on positives where the
     organ happens to be cropped out; that's the existing CT-track
     convention, kept as-is here for consistency).

Anatomy-wise ground truth is the SAME per-image global label used for the
whole-image test (e.g. "heart_cardiomegaly" and "cardiomegaly" both check
the same labels_raw membership) -- calc_metrics.py does this too
(`all_labels[disease]` is looked up once per test_item regardless of which
organ's image feature is being scored against it), it does not have or
need a separate per-organ ground truth signal.

Usage:
    cd RADAR/X-ray/phase6
    python3 -u zero_shot_eval.py [--limit N]
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# --- bypass lavis/__init__.py's heavy import chain (same trick as phase5/train.py) ---
_PHASE3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3")
_LAVIS_DIR = os.path.join(_PHASE3_DIR, "lavis")
sys.path.insert(0, _PHASE3_DIR)

_fake_lavis = types.ModuleType("lavis")
_fake_lavis.__path__ = [os.path.abspath(_LAVIS_DIR)]
sys.modules["lavis"] = _fake_lavis

from lavis.common.registry import registry  # noqa: E402
registry.register_path("library_root", os.path.abspath(_LAVIS_DIR))

# --- transformers>=4.28 moved/removed these from transformers.modeling_utils
# (med.py was vendored against transformers==4.25, which had them there).
# apply_chunking_to_forward/prune_linear_layer just moved to pytorch_utils;
# find_pruneable_heads_and_indices was removed outright, but it's only
# reachable via BertSelfAttention.prune_heads(), which this RADAR pipeline
# never calls (no head pruning) -- it only needs to exist at import time,
# so it's vendored back in verbatim from pre-4.28 transformers.
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

# med.py's BertModel/BertEmbeddings/etc. subclasses predate transformers'
# post_init() convention (they call the older init_weights() directly), so
# all_tied_weights_keys/_keep_in_fp32_modules/etc. -- which post_init() sets
# and which newer from_pretrained()'s _finalize_model_loading() now assumes
# exist -- never get set. Call post_init() ourselves right before the
# original finalize step runs, only when it's actually missing.
_orig_finalize_model_loading = _hf_modeling_utils.PreTrainedModel._finalize_model_loading


@staticmethod
def _patched_finalize_model_loading(model, load_config, loading_info):
    if not hasattr(model, "all_tied_weights_keys"):
        model.post_init()
    return _orig_finalize_model_loading(model, load_config, loading_info)


_hf_modeling_utils.PreTrainedModel._finalize_model_loading = _patched_finalize_model_loading

# ModuleUtilsMixin's get_head_mask/get_extended_attention_mask/invert_attention_mask
# were removed entirely in this transformers version (models now build attention
# bias masks internally, e.g. for SDPA), but med.py's BertModel.forward() (a
# pre-refactor vendored copy) still calls all three by name on `self`. These
# implementations were stable/unchanged across transformers for years -- vendored
# back in verbatim rather than rewriting med.py's forward() to not need them.
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
from dataset import (  # noqa: E402
    TAR_PATH, CHEXMASK_MANIFEST_PATH, CAPTIONS_MANIFEST_PATH, SPLIT_PATH,
    IMG_SIZE, ORGANS, percentile_normalize_uint8,
)

import io
import tarfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt", "checkpoint_radar_pretrain_xray.pth")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# finding -> which organ(s) to also test anatomy-wise against (in addition
# to always being tested whole-image). Picked from phase2/region_rules.py's
# own HEART_LABEL_KEYWORDS/loc vocabulary (heart findings) and from val-set
# label frequency (>= ~40 positives -- see phase6/README.md) for lung
# findings tested against both lungs (PadChest's Labels field doesn't
# reliably give per-image laterality, so both lungs share the same global
# ground truth, exactly like calc_metrics.py's per-organ test_items sharing
# one disease-level label).
HEART_FINDINGS = ["cardiomegaly", "aortic elongation", "pacemaker", "sternotomy"]
LUNG_FINDINGS = [
    "copd signs", "pneumonia", "pleural effusion", "infiltrates", "nodule",
    "interstitial pattern", "costophrenic angle blunting", "fibrotic band",
    "laminar atelectasis", "bronchiectasis",
]
WHOLE_ONLY_FINDINGS = [
    "hiatal hernia", "scoliosis", "kyphosis", "calcified granuloma",
    "vertebral degenerative changes", "apical pleural thickening",
]
ALL_WHOLE_FINDINGS = HEART_FINDINGS + LUNG_FINDINGS + WHOLE_ONLY_FINDINGS


def _fix_position_ids_buffer(bert_module):
    # transformers' from_pretrained() constructs modules under a meta-device
    # "fast init" context; med.py's BertEmbeddings.__init__ registers
    # position_ids via torch.arange(...) at construction time, but under
    # fast-init that arange never gets materialized, leaving garbage values
    # -- masked here only because the strict=True load below happens to also
    # restore it from the checkpoint (which was trained before this
    # environment reset). Fixed explicitly anyway so this doesn't silently
    # depend on that side effect (see phase5/train.py's build_model() for
    # the case -- fresh pretraining -- where no checkpoint saves it).
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


class ZeroShotEvalDataset(Dataset):
    """Val-split images only, deterministic (no flips) -- same tar/preprocessing
    as phase4/dataset.py's RadarXrayDataset, but also carries labels_raw for
    ground truth and skips loading masks/captions entirely (forward_test_win
    uses the model's OWN predicted segmentation, not CheXmask)."""

    def __init__(self, split="val", limit=None, seed=42):
        with open(SPLIT_PATH) as f:
            split_ids = set(json.load(f)[f"{split}_image_ids"])

        quality_ok = {}
        with open(CHEXMASK_MANIFEST_PATH) as f:
            for line in f:
                rec = json.loads(line)
                quality_ok[rec["image_id"]] = rec.get("quality_ok", False)

        self.records = []
        with open(CAPTIONS_MANIFEST_PATH) as f:
            for line in f:
                rec = json.loads(line)
                image_id = rec["image_id"]
                if image_id not in split_ids or not quality_ok.get(image_id, False):
                    continue
                labels_lower = {lab.lower() for lab in rec["labels_raw"]}
                self.records.append({"image_id": image_id, "labels": labels_lower})

        if limit and limit < len(self.records):
            # random subsample (not a plain head-slice) so a smaller --limit
            # on the much bigger train split isn't biased by manifest order
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.records), size=limit, replace=False)
            self.records = [self.records[i] for i in idx]
        print(f"ZeroShotEvalDataset[{split}]: {len(self.records)} records")

        self._tar = None
        self._members_by_name = None

    def _ensure_tar_open(self):
        if self._tar is None:
            self._tar = tarfile.open(TAR_PATH)
            self._members_by_name = {os.path.basename(m.name): m for m in self._tar.getmembers()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self._ensure_tar_open()
        rec = self.records[index]
        member = self._members_by_name[rec["image_id"]]
        with self._tar.extractfile(member) as fh:
            raw = np.array(Image.open(io.BytesIO(fh.read())))
        img8 = percentile_normalize_uint8(raw)
        img = np.array(Image.fromarray(img8).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        image = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        return {"image": image, "image_id": rec["image_id"]}


@torch.inference_mode()
def build_text_feat_dict(model, findings, device):
    """finding -> (2, embed_dim) text feat: row 0 = 'normal.' (negative),
    row 1 = '<finding>.' (positive). Matches phase2's own caption style
    (concatenated '<label>.' tokens, 'normal.' for clean images) -- the
    closest in-distribution text to what the model was actually trained on."""
    text_feat_dict = {}
    for finding in findings:
        text = model.tokenizer(
            ["normal.", f"{finding}."],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
        text_feat_dict[finding] = text_feat
    return text_feat_dict


@torch.inference_mode()
def whole_image_pass(model, loader, device, findings, text_feat_dict):
    scores = {finding: {} for finding in findings}
    for batch in loader:
        image = batch["image"].to(device)
        image_id = batch["image_id"][0]

        _, _, embeds1, embeds2, embeds3, _, _, _ = model.visual_encoder(image, None)
        key_whole = value_whole = torch.cat([embeds1, embeds2, embeds3], dim=1)
        query_whole = model.query_tokens_whole.unsqueeze(0).expand(key_whole.shape[0], -1, -1)
        image_feat_whole, _ = model.attention_whole(query_whole, key_whole, value_whole)
        image_feat_whole = image_feat_whole.squeeze(1)
        image_feat_whole = F.normalize(model.vision_projs_whole(image_feat_whole), dim=-1)

        for finding in findings:
            text_feat = text_feat_dict[finding]
            logits = image_feat_whole @ text_feat.t() / model.temp
            prob_positive = logits.softmax(-1)[0, 1].item()
            scores[finding][image_id] = prob_positive
    return scores


@torch.inference_mode()
def anatomy_pass(model, loader, device, test_items):
    """test_items: list of (organ, finding) pairs. Returns
    {(organ, finding): {image_id: prob_positive}}, using 0.0 for images
    where that organ isn't intact/visible (calc_metrics.py's convention)."""
    prompt_items = [(organ, finding, "normal.", f"{finding}.") for organ, finding in test_items]
    text_feat_dict = model.prepare_text_feat(prompt_items, length=64)

    scores = {item: {} for item in test_items}
    for batch in loader:
        image = batch["image"].to(device)
        image_id = batch["image_id"][0]

        organ_logits = {tuple(item): [] for item in prompt_items}
        organ_feat_dict = {}
        model.forward_test_win(
            images=image,
            masks=None,
            organ_logits=organ_logits,
            test_organs=ORGANS,
            text_feat_dict=text_feat_dict,
            organ_feat_dict=organ_feat_dict,
            whole_organ_sizes=None,
        )

        for prompt_item, (organ, finding) in zip(prompt_items, test_items):
            result = organ_logits[tuple(prompt_item)]
            prob_positive = result[0][0][1] if result else 0.0
            scores[(organ, finding)][image_id] = prob_positive
    return scores


def compute_aucs(scores_by_key, gt_by_image_id, label_key_fn):
    """scores_by_key: {key: {image_id: prob}}. label_key_fn(key) -> finding
    name to look up in gt_by_image_id. Returns {key: auc}."""
    aucs = {}
    for key, image_scores in scores_by_key.items():
        finding = label_key_fn(key)
        gt_labels, pd_scores = [], []
        for image_id, prob in image_scores.items():
            gt_labels.append(1.0 if finding in gt_by_image_id[image_id] else 0.0)
            pd_scores.append(prob)
        if len(set(gt_labels)) < 2:
            print(f"  skip {key}: only one class present ({len(gt_labels)} images)")
            continue
        aucs[key] = roc_auc_score(gt_labels, pd_scores)
    return aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="val",
                         help="train: underfit/overfit diagnostic (does the model even fit what it trained on); val: the real zero-shot generalization number")
    parser.add_argument("--limit", type=int, default=None, help="cap dataset size for a quick dry run, or for a train-split subsample comparable in size to val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(device)
    ds = ZeroShotEvalDataset(split=args.split, limit=args.limit)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    gt_by_image_id = {rec["image_id"]: rec["labels"] for rec in ds.records}

    print(f"\n=== whole-image zero-shot ({len(ALL_WHOLE_FINDINGS)} findings) ===")
    text_feat_dict_whole = build_text_feat_dict(model, ALL_WHOLE_FINDINGS, device)
    whole_scores = whole_image_pass(model, loader, device, ALL_WHOLE_FINDINGS, text_feat_dict_whole)
    whole_aucs = compute_aucs(whole_scores, gt_by_image_id, label_key_fn=lambda k: k)
    for finding, auc in sorted(whole_aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {finding:35s} AUC={auc:.4f}")
    if whole_aucs:
        print(f"  --> AvgAUC_whole: {np.mean(list(whole_aucs.values())):.4f}")

    anatomy_test_items = (
        [("heart", f) for f in HEART_FINDINGS]
        + [("left_lung", f) for f in LUNG_FINDINGS]
        + [("right_lung", f) for f in LUNG_FINDINGS]
    )
    print(f"\n=== anatomy-wise zero-shot ({len(anatomy_test_items)} organ/finding pairs) ===")
    anatomy_scores = anatomy_pass(model, loader, device, anatomy_test_items)
    anatomy_aucs = compute_aucs(anatomy_scores, gt_by_image_id, label_key_fn=lambda k: k[1])
    for (organ, finding), auc in sorted(anatomy_aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {organ:10s} {finding:35s} AUC={auc:.4f}")
    if anatomy_aucs:
        print(f"  --> AvgAUC_anatomy: {np.mean(list(anatomy_aucs.values())):.4f}")

    all_aucs = list(whole_aucs.values()) + list(anatomy_aucs.values())
    if all_aucs:
        print(f"\n--> AvgAUC (all): {np.mean(all_aucs):.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"xray_zero_shot_aucs_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump({
            "whole": {k: float(v) for k, v in whole_aucs.items()},
            "anatomy": {f"{k[0]}_{k[1]}": float(v) for k, v in anatomy_aucs.items()},
        }, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
