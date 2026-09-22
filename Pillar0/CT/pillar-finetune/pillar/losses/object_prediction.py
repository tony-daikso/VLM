from collections import OrderedDict

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from pillar.losses.abstract import AbstractLoss
from pillar.utils.detr import (
    get_world_size,
    is_dist_avail_and_initialized,
    generalized_box_iou_3d,
    box_cxcyczwhd_to_xyzxyz,
    mask_to_bounding_box,
)
from pillar.utils.matcher import HungarianMatcher3D


class RegionAnnotationLoss(AbstractLoss):
    def __init__(self, args, attn_map_key="volume_attention_map", **kwargs):
        super().__init__(args)
        self.attn_map_key = attn_map_key
        self.nested_key = kwargs.get("nested_key", "")

    def __call__(self, batch=None, model_output=None, **extras):
        # Initialize loss and logging dict
        batch_mask = batch["has_annotation"].bool()
        attention_loss = torch.tensor(0.0, device=batch_mask.device)
        logging_dict = OrderedDict()

        # If there are any elements with annotations, compute the loss
        if batch_mask.sum() > 0:
            if self.nested_key:
                pred_attention = model_output[self.nested_key][self.attn_map_key]
            else:
                pred_attention = model_output[self.attn_map_key]
            assert pred_attention.shape[1] == 1, "Only implemented for 1 query right now"
            if pred_attention is not None:
                annotation_gold = batch["image_annotations"].float() * batch_mask[:, None, None, None, None]

                # Reshaping to match prediction
                mask_size = pred_attention.shape[2:]
                annotation_gold = F.interpolate(annotation_gold, size=mask_size, mode="trilinear", align_corners=False)

                attention_product = (pred_attention * annotation_gold).sum(dim=(2, 3, 4))  # [B, 1]
                attention_product = torch.clamp(attention_product, min=1e-6)  # for numerical stability
                attention_loss = -torch.log(attention_product[batch_mask])
                attention_loss = attention_loss.mean()

        logging_dict[f"region_annotation_loss_{self.attn_map_key}"] = attention_loss.detach()

        return attention_loss, logging_dict

    @property
    def loss_keys(self):
        return {"target_label": ["image_annotations", "has_annotation"], "pred_label": self.attn_map_key}


class SybilRegionAnnotationLoss(AbstractLoss):
    def __init__(self, args, image_attention_loss_lambda, volume_attention_loss_lambda):
        super().__init__(args)
        self.image_attention_loss_lambda = image_attention_loss_lambda
        self.volume_attention_loss_lambda = volume_attention_loss_lambda

    def __call__(self, batch=None, model_output=None, **extras):
        total_loss, logging_dict = 0, OrderedDict()

        (
            B,
            _,
            N,
            H,
            W,
        ) = model_output["activ"].shape

        batch_mask = batch["has_annotation"].bool()

        for attn_num in [1, 2]:
            side_attn = -1
            if model_output.get("image_attention_{}".format(attn_num), None) is not None:
                if len(batch["image_annotations"].shape) == 4:
                    batch["image_annotations"] = batch["image_annotations"].unsqueeze(1)
                else:
                    batch["image_annotations"] = batch["image_annotations"][:, :1, ...]

                # resize annotation to 'activ' size
                # annotation_gold: [B, C, N, H, W]
                annotation_gold = F.interpolate(batch["image_annotations"], (N, H, W), mode="area")
                annotation_gold = annotation_gold * batch_mask[:, None, None, None, None]

                # renormalize scores
                mask_area = annotation_gold.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
                mask_area[mask_area == 0] = 1
                annotation_gold /= mask_area

                # reshape annotation into 1D vector
                annotation_gold = annotation_gold.view(B, N, -1).float()

                # get mask over annotation boxes in order to weigh
                # non-annotated scores with zero when computing loss
                annotation_gold_mask = (annotation_gold > 0).float()

                num_annotated_samples = (annotation_gold.view(B * N, -1).sum(-1) > 0).sum()
                # num_annotated_samples = max(1, num_annotated_samples)
                if num_annotated_samples == 0:
                    num_annotated_samples = 1

                # pred_attn will be 0 for volumes without annotation
                pred_attn = model_output["image_attention_{}".format(attn_num)] * batch_mask[:, None, None]
                kldiv = F.kl_div(pred_attn, annotation_gold, reduction="none") * annotation_gold_mask

                # sum loss per volume and average over batches
                loss = kldiv.sum() / num_annotated_samples
                logging_dict["image_attention_loss_{}".format(attn_num)] = loss.detach()
                total_loss += self.image_attention_loss_lambda * loss

                # attend to cancer side
                cancer_side_mask = (batch["cancer_laterality"][:, :2].sum(-1) == 1).float()[
                    :, None
                ]  # only one side is positive
                cancer_side_gold = (
                    batch["cancer_laterality"][:, 1].unsqueeze(1).repeat(1, N)
                )  # left side (seen as lung on right) is positive class
                # num_annotated_samples = max(N * cancer_side_mask.sum(), 1)
                num_annotated_samples = N * cancer_side_mask.sum()
                if num_annotated_samples == 0:
                    num_annotated_samples += 1

                side_attn = torch.exp(model_output["image_attention_{}".format(attn_num)])
                side_attn = side_attn.view(B, N, H, W)
                side_attn = torch.stack(
                    [
                        side_attn[:, :, :, : W // 2].sum((2, 3)),
                        side_attn[:, :, :, W // 2 :].sum((2, 3)),
                    ],
                    dim=-1,
                )
                side_attn_log = F.log_softmax(side_attn, dim=-1).transpose(1, 2)

                loss = (
                    F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none") * cancer_side_mask
                ).sum() / num_annotated_samples
                logging_dict["image_side_attention_loss_{}".format(attn_num)] = loss.detach()
                total_loss += self.image_attention_loss_lambda * loss

            if model_output.get("volume_attention_{}".format(attn_num), None) is not None:
                # find size of annotation box per slice and normalize
                annotation_gold = batch["annotation_areas"].float() * batch_mask[:, None]

                # Use `num_images` in the dataset
                if N != self.args.dataset.shared_dataset_kwargs.num_images:
                    annotation_gold = F.interpolate(
                        annotation_gold.unsqueeze(1),
                        (N),
                        mode="linear",
                        align_corners=True,
                    )[:, 0]
                area_per_slice = annotation_gold.sum(-1).unsqueeze(-1)
                area_per_slice[area_per_slice == 0] = 1
                annotation_gold /= area_per_slice

                num_annotated_samples = (annotation_gold.sum(-1) > 0).sum()
                num_annotated_samples = max(1, num_annotated_samples)

                # find slices with annotation
                annotation_gold_mask = (annotation_gold > 0).float()

                pred_attn = model_output["volume_attention_{}".format(attn_num)] * batch_mask[:, None]
                kldiv = F.kl_div(pred_attn, annotation_gold, reduction="none") * annotation_gold_mask  # B, N
                loss = kldiv.sum() / num_annotated_samples

                logging_dict["volume_attention_loss_{}".format(attn_num)] = loss.detach()
                total_loss += self.volume_attention_loss_lambda * loss

                if isinstance(side_attn, torch.Tensor):
                    # attend to cancer side
                    cancer_side_mask = (
                        batch["cancer_laterality"][:, :2].sum(-1) == 1
                    ).float()  # only one side is positive
                    cancer_side_gold = batch["cancer_laterality"][
                        :, 1
                    ]  # left side (seen as lung on right) is positive class
                    num_annotated_samples = max(cancer_side_mask.sum(), 1)

                    pred_attn = torch.exp(model_output["volume_attention_{}".format(attn_num)])
                    side_attn = (side_attn * pred_attn.unsqueeze(-1)).sum(1)
                    side_attn_log = F.log_softmax(side_attn, dim=-1)

                    loss = (
                        F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none") * cancer_side_mask
                    ).sum() / num_annotated_samples
                    logging_dict["volume_side_attention_loss_{}".format(attn_num)] = loss.detach()
                    total_loss += self.volume_attention_loss_lambda * loss

        return total_loss, logging_dict

    @property
    def loss_keys(self):
        return {
            "target_label": ["image_annotations", "has_annotation", "cancer_laterality"],
            "pred_label": [
                "activ",
                "image_attention_1",
                "image_attention_2",
                "volume_attention_1",
                "volume_attention_2",
            ],
        }


# Modified from: https://github.com/facebookresearch/detr for 3D use + compatible with pillar

class DETRObjectDetectionLoss(AbstractLoss):
    def __init__(
        self,
        args,
        num_classes=None,
        cost_class: float = 1,
        cost_bbox: float = 5,
        cost_giou: float = 2,
        eos_coef: float = 0.1,
        **kwargs,
    ):
        if num_classes:
            self.num_classes = num_classes
        else:
            warnings.warn("Infering num_classes from args.model.kwargs.head_kwargs.num_classes")
            self.num_classes = args.model.kwargs.head_kwargs.num_classes
        matcher = HungarianMatcher3D(cost_class, cost_bbox, cost_giou)
        weight_dict = {"loss_ce": cost_class, "loss_bbox": cost_bbox, "loss_giou": cost_giou}
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
        )
        self.nested_key = kwargs.get("nested_key", "")

    def __call__(self, batch=None, model_output=None, **extras):
        # See if any of the elements has annotations in this batch
        batch_mask = batch["has_annotation"].bool()
        outputs = OrderedDict()
        loss = torch.tensor(0.0, device=batch_mask.device)
        logging_dict = OrderedDict()

        if batch_mask.sum() > 0:
            targets = []
            valid_indices = []  # Track which batch indices have valid annotations
            for batch_idx in range(batch["image_annotations"].shape[0]):
                if batch_mask[batch_idx]:  # Process only valid elements
                    annotations = batch["image_annotations"][batch_idx]

                    # Check that the annotations contain actual foreground pixels
                    if annotations.sum() == 0:
                        continue  # Skip if no foreground pixels

                    if annotations.shape[0] != 1:  # Ensure only one object per image
                        print("More than one object in the image")
                        continue  # Skip if more than one object

                    # Convert mask to bounding box
                    box = mask_to_bounding_box(annotations[0])

                    # Add the target for this image
                    targets.append(
                        {
                            "boxes": box.unsqueeze(0),  # Shape [1, 6]
                            "labels": torch.tensor([1], dtype=torch.long),  # Shape [1]
                        }
                    )
                    valid_indices.append(batch_idx)

            # Skip loss calculation if no valid targets after filtering
            if len(targets) == 0:
                logging_dict["total_bbox_loss"] = torch.tensor(0.0, device=batch_mask.device)
                return loss, logging_dict

            # Get the outputs only for items that have valid annotations
            valid_indices = torch.tensor(valid_indices, device=batch_mask.device)
            if self.nested_key:
                # Access the nested dictionary outputs
                detr_outputs = model_output[self.nested_key]
                outputs["pred_boxes"] = detr_outputs["pred_boxes"][valid_indices]
                outputs["pred_logits"] = detr_outputs["pred_logits"][valid_indices]
            else:
                outputs["pred_boxes"] = model_output["pred_boxes"][valid_indices]
                outputs["pred_logits"] = model_output["pred_logits"][valid_indices]

            # Move targets to the same device as model outputs
            device = outputs["pred_boxes"].device
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            # Compute the loss
            try:
                criterion_results = self.criterion(outputs, targets)
                logging_dict["loss_ce"] = criterion_results["loss_ce"].detach()
                logging_dict["loss_bbox"] = criterion_results["loss_bbox"].detach()
                logging_dict["loss_giou"] = criterion_results["loss_giou"].detach()
                logging_dict["total_bbox_loss"] = criterion_results["total_loss"].detach()
                loss = criterion_results["total_loss"]
            except Exception as e:
                print(batch["sample_name"])
                print(e)
                loss = torch.tensor(0.0)
                logging_dict["total_bbox_loss"] = torch.tensor(0.0)
        return loss, logging_dict

    @property
    def loss_keys(self):
        return {"target_label": ["image_annotations", "has_annotation"], "pred_label": ["pred_boxes", "pred_logits"]}


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        device = src_logits.device

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o

        # Make sure empty_weight is on the same device
        empty_weight = self.empty_weight.to(device)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou_3d(
                box_cxcyczwhd_to_xyzxyz(src_boxes),  # Convert to corner format for IoU
                box_cxcyczwhd_to_xyzxyz(target_boxes),
            )
        )  # updated for 3D
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=outputs["pred_boxes"].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             if loss == 'masks':
        #                 # Intermediate masks losses are too costly to compute, we ignore them.
        #                 continue
        #             kwargs = {}
        #             if loss == 'labels':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        losses["total_loss"] = total_loss

        return losses
