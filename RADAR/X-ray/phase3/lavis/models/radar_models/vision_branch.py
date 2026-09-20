
from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import sys

from monai.utils import deprecated_arg
import os
import pydoc
import warnings
from typing import Union
import torch.nn.functional as F

# Same convention as the CT track's checkpoint_unet.pth: a plain path
# relative to cwd, assuming scripts are run from within this directory
# (RADAR/X-ray/phase3/), so '../ckpt' means RADAR/X-ray/ckpt/. Produced by
# RADAR/X-ray/phase3/pretrain_segmentation.py (trains just self.UNet on
# CheXmask masks with a Dice loss, mirroring how the CT track separately
# pretrains its UNet on TotalSegmentator masks before RADAR proper loads
# `checkpoint_unet.pth`). See RADAR/X-ray/PLAN.md Phase 3.
XRAY_UNET_CKPT_PATH = "../ckpt/checkpoint_unet_xray.pth"

__all__ = ["VisionBranch"]


class VisionBranch(nn.Module):
    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels=1
    ) -> None:
        super().__init__()
        
        # define UNet (2D port of the CT track's 3D UNet -- see
        # RADAR/X-ray/PLAN.md Phase 3. dynamic_network_architectures'
        # PlainConvUNetLightD is dimension-agnostic; only conv_op/norm_op and
        # the kernel/stride tuples change from 3D to 2D, everything else
        # (n_stages, features_per_stage) is kept identical to the CT config.
        # Downsampling schedule: stage 0 keeps full resolution, stages 1-5
        # each halve H,W -- cumulative 2x/4x/8x/16x/32x, matching the
        # in-plane (H,W) downsampling the CT config used (the CT config's
        # extra stage-0/1 asymmetry only existed to handle anisotropic CT
        # slice spacing along D, which doesn't apply to a single 2D image).
        self.UNet = self.get_network_from_plans(
            arch_class_name="dynamic_network_architectures.architectures.unet_lightdecoder.PlainConvUNetLightD",
            arch_kwargs={
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "conv_op": "torch.nn.modules.conv.Conv2d",
                "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.BatchNorm2d",
                "norm_op_kwargs": {},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.ReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=1,
            output_channels=len(self._organs()) + 1,
            allow_init=True,
            deep_supervision=True,
        )

        # Load the X-ray segmentation-pretraining checkpoint if it exists
        # (produced by pretrain_segmentation.py). The CT checkpoint itself
        # can't be reused here -- a 2D UNet's conv weights are rank-4,
        # incompatible with the CT checkpoint's rank-5 Conv3d weights.
        # Optional rather than required: until pretrain_segmentation.py has
        # actually been run (it needs Phase 1's CheXmask masks), RADAR
        # pretraining still works, just starting the UNet from scratch.
        if os.path.exists(XRAY_UNET_CKPT_PATH):
            checkpoint = torch.load(XRAY_UNET_CKPT_PATH, map_location=torch.device('cpu'), weights_only=False)
            msg = self.UNet.load_state_dict(checkpoint["network_weights"], strict=False)
            print(f"[VisionBranch] loaded {XRAY_UNET_CKPT_PATH}: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
        else:
            warnings.warn(
                f"[VisionBranch] {XRAY_UNET_CKPT_PATH} not found -- UNet starts from random init "
                "(run pretrain_segmentation.py first for a segmentation-pretrained start)."
            )

        self.proj1 = nn.Conv2d(320, 256, kernel_size=1)
        self.proj2 = nn.Conv2d(320, 256, kernel_size=1)
        self.proj3 = nn.Conv2d(256, 256, kernel_size=1)

        self.organs = self._organs()

    @staticmethod
    def _organs():
        # CT used 36 abdominal organs from TotalSegmentator; CheXmask only
        # provides 3 CXR anatomical masks (heart doubles as heart/mediastinum
        # since CheXmask has no separate mediastinum mask). See
        # RADAR/X-ray/PLAN.md Phase 1/3 and RADAR/X-ray/phase2/region_rules.py,
        # which use these same 3 region names.
        return ["left_lung", "right_lung", "heart"]

    def get_network_from_plans(sefl, arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
        network_class = arch_class_name
        architecture_kwargs = dict(**arch_kwargs)
        for ri in arch_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        nw_class = pydoc.locate(network_class)

        if deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = deep_supervision

        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize') and allow_init:
            network.apply(network.initialize)

        return network

    def forward(self, x, y):
        skips, segs = self.UNet(x)
        
        scale1 = self.proj1(skips[-1])
        scale2 = self.proj2(skips[-2])
        scale3 = self.proj3(skips[-3])
        
        # process mask (2D: bilinear instead of trilinear, 2-element target
        # size instead of 3-element -- see vision_branch.py port notes above)
        pred_logit = segs[0]
        seg_probs = torch.softmax(pred_logit, 1)
        target_size = [pred_logit.shape[-2]*2, pred_logit.shape[-1]*2]
        pred_logit = F.interpolate(pred_logit, size=target_size, mode='bilinear', align_corners=False)
        pred_mask = torch.softmax(pred_logit, 1)
        pred_mask = pred_mask.argmax(1)
        y = pred_mask
        
        res_x1 = scale1.flatten(2).transpose(1, 2)
        res_x2 = scale2.flatten(2).transpose(1, 2)
        res_x3 = scale3.flatten(2).transpose(1, 2)
        
        B, L1, _ = res_x1.size()
        B, L2, _ = res_x2.size()
        B, L3, _ = res_x3.size()
        
        with torch.no_grad():
            organ_token_flags1 = torch.zeros(B, len(self.organs), L1, dtype=bool).to(x.device)
            organ_token_flags2 = torch.zeros(B, len(self.organs), L2, dtype=bool).to(x.device)
            organ_token_flags3 = torch.zeros(B, len(self.organs), L3, dtype=bool).to(x.device)

            b = x.size(0)
            for i in range(b):
                unique_values = torch.unique(y[i])
                unique_values = unique_values[unique_values != 0]
                if unique_values.tolist() == []:
                    continue
                masks = torch.stack([torch.eq(y[i], uv) for uv in unique_values]).float()

                # 2D: drop the depth-axis factor from each CT kernel/stride
                # (2,8,8)/(4,16,16)/(8,32,32) -> (8,8)/(16,16)/(32,32),
                # matching this port's H,W-only downsampling schedule (see
                # __init__ port notes above)
                highlight_tokens3 = F.max_pool2d(
                    masks.unsqueeze(1),
                    kernel_size=(8, 8),
                    stride=(8, 8)
                ).flatten(1) > 0

                highlight_tokens2 = F.max_pool2d(
                    masks.unsqueeze(1),
                    kernel_size=(16, 16),
                    stride=(16, 16)
                ).flatten(1) > 0

                highlight_tokens1 = F.max_pool2d(
                    masks.unsqueeze(1),
                    kernel_size=(32, 32),
                    stride=(32, 32)
                ).flatten(1) > 0

                organ_token_flags1[i][unique_values.long() - 1] = highlight_tokens1 > 0
                organ_token_flags2[i][unique_values.long() - 1] = highlight_tokens2 > 0
                organ_token_flags3[i][unique_values.long() - 1] = highlight_tokens3 > 0
        return seg_probs, pred_mask, res_x1, res_x2, res_x3, organ_token_flags1, organ_token_flags2, organ_token_flags3

