"""
Engine base class for distributed training, evaluation.
"""

import os
from collections import OrderedDict
from typing import Literal

import torch
import wandb
import pandas as pd

from pillar import losses, metrics
from pillar.utils.logging import logger
from pillar.utils.engine import gather_step_outputs
from pillar.utils.misc import rank_zero_only, get_is_master

Split = Literal["train", "val", "test"]


def parse_amp_precision(precision):
    parse = {
        "bf16-mixed": torch.bfloat16,
        "fp16-mixed": torch.float16,
        "32": None,  # disable AMP
    }

    return parse[str(precision)]


class Engine(object):
    def __init__(
        self,
        args,
        *,
        accumulate_grad_batches=1,
        resume,
        max_epochs,
        precision,
        clip_grad=None,
        log_grad_norm=False,
        dataset_info=None,
        limit_num_batches=None,
        use_gpu_augs=False,
        train_gpu_augs=[],
        test_gpu_augs=[],
        **kwargs,
    ):
        self.args = args
        self.accum_iter = accumulate_grad_batches
        self.training_step_outputs = []
        self.dataset_info = dataset_info
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.use_gpu_augs = use_gpu_augs
        self.train_gpu_augs = train_gpu_augs
        self.test_gpu_augs = test_gpu_augs

        self.global_step = 0
        self.amp_precision = parse_amp_precision(precision)

        self.limit_num_batches = limit_num_batches

        # Cache step metrics to avoid recreating them on every batch
        self._step_metrics = None

        # resume and max_epochs are handled outside of the engine

        if kwargs:
            logger.warning(f"Ignoring unrecognized kwargs to engine: {kwargs}")

    @rank_zero_only
    def save_on_master(self, ckpt_dir=None, epoch=0, state=None):
        # Outside of the engine (in main) we call this function only with master. Here we still use `rank_zero_only` to be safe.
        os.makedirs(ckpt_dir, exist_ok=True)
        if epoch == -1:
            path = f"{ckpt_dir}/latest.ckpt"
        else:
            path = f"{ckpt_dir}/epoch={epoch}.ckpt"
        torch.save(state, path)

    def load(self, path, map_location="cpu"):
        return torch.load(path, map_location=map_location, weights_only=False)

    def get_epoch_metrics(self, split: Split):
        if "epoch_metrics" not in self.args.metrics or split not in self.args.metrics.epoch_metrics:
            return []
        epoch_metrics = []
        for metric_info in self.args.metrics.epoch_metrics[split]:
            name = metric_info["type"]
            kwargs = metric_info["kwargs"] if "kwargs" in metric_info else {}
            kwargs["split"] = split
            # Survivor metric requires training dataset to compute
            metric = metrics.__dict__[name](self.args, dataset_info=self.dataset_info, **kwargs)
            epoch_metrics.append(metric)
        return epoch_metrics

    def compute_metrics(self, metric_input):
        if "step_metrics" not in self.args.metrics:
            return []

        # Initialize step metrics once and cache them
        if self._step_metrics is None:
            self._step_metrics = []
            for metric_info in self.args.metrics.step_metrics:
                name = metric_info["type"]
                kwargs = metric_info["kwargs"] if "kwargs" in metric_info else {}
                metric_fn = metrics.__dict__[name](self.args, **kwargs)
                self._step_metrics.append(metric_fn)

        metric_dict = OrderedDict()
        for metric_fn in self._step_metrics:
            local_metric_dict = metric_fn(**metric_input)
            metric_dict.update(local_metric_dict)
        return metric_dict

    def compute_step_metrics(self, loss_input: dict, metric_input: dict, train: bool):
        logging_dict = OrderedDict()
        device = next(iter(loss_input["model_output"].values())).device
        # loss = torch.tensor(0.0, device=device)

        # if train != False:
        loss, loss_dict = self.compute_loss(loss_input, train)
        logging_dict.update(loss_dict)

        if metric_input is not None:
            metric_dict = self.compute_metrics(metric_input)
            logging_dict.update(metric_dict)

        return loss, logging_dict

    def compute_epoch_metrics(self, result_dict, args, device, *, key_prefix, epoch, split, epoch_metrics):
        stats_dict = OrderedDict()
        for metric in epoch_metrics:
            metric_stats = metric(**result_dict)
            for k, v in metric_stats.items():
                if isinstance(v, torch.Tensor):
                    stats_dict[key_prefix + k] = v.to(device)
                else:
                    stats_dict[key_prefix + k] = torch.tensor(v)
        return stats_dict

    def get_losses(self):
        if not hasattr(self.args.metrics, "losses") or self.args.metrics.losses is None:
            return None
        assert len(self.args.metrics.losses) > 0, "Must specify at least one loss function in args.metrics.losses"
        loss_functions = []
        for loss_info in self.args.metrics.losses:
            name = loss_info["type"]
            kwargs = loss_info["kwargs"] if "kwargs" in loss_info else {}
            weight = loss_info["weight"]
            enable_at_train = loss_info.get("enable_at_train", True)
            enable_at_eval = loss_info.get("enable_at_eval", True)
            fn = losses.__dict__[name](self.args, **kwargs)
            loss_functions.append((fn, name, weight, enable_at_train, enable_at_eval))
        return loss_functions

    def compute_loss(self, loss_input, train):
        total_loss = 0
        losses = self.get_losses()
        loss_dict = OrderedDict()
        if losses is None:
            logger.warning(f"losses are not configured ({losses}), returning 0")
            total_loss = torch.tensor([0.0])
            return total_loss, loss_dict

        for loss_fn, _, weight, enable_at_train, enable_at_eval in losses:
            if train:
                if not enable_at_train:
                    continue
            else:
                if not enable_at_eval:
                    continue
            loss, local_loss_dict = loss_fn(**loss_input)
            total_loss += weight * loss
            loss_dict.update(local_loss_dict)
        return total_loss, loss_dict

    def step(self, batch, batch_idx, device=None, split="train"):
        raise NotImplementedError

    def on_epoch_end(self, split="train", device="cuda", epoch=0, metadata=None, ckpt_dir=None):
        epoch_metrics = self.get_epoch_metrics(split)
        # We don't need to gather step outputs when there are no metrics defined.
        if len(epoch_metrics) == 0:
            if split == "train":
                self.training_step_outputs.clear()
            elif split == "val":
                self.validation_step_outputs.clear()
            elif split == "test":
                self.test_step_outputs.clear()
            return
        if split == "train":
            outputs = gather_step_outputs(self.training_step_outputs)
            self.training_step_outputs.clear()
        elif split == "val":
            outputs = gather_step_outputs(self.validation_step_outputs)
            self.validation_step_outputs.clear()
        elif split == "test":
            outputs = gather_step_outputs(self.test_step_outputs)
            self.test_step_outputs.clear()

        if len(outputs) == 0:
            return
        if ckpt_dir is not None:
            os.makedirs(f"{ckpt_dir}/{epoch}", exist_ok=True)
            logger.info(f"Saving {split} outputs to {ckpt_dir}/{epoch}/{split}.pth")
            torch.save(outputs, f"{ckpt_dir}/{epoch}/{split}.pth")

            output_dict = {
                key: [str(x) for x in value.cpu().numpy()] if isinstance(value, torch.Tensor) else value
                for key, value in outputs.items()
                if key != "logs"
            }
            pd.DataFrame(output_dict).to_csv(f"{ckpt_dir}/{epoch}/{split}.csv", index=False)

        epoch_metrics = self.compute_epoch_metrics(
            outputs,
            self.args,
            device,
            key_prefix=f"{split}_",
            epoch=epoch,
            split=split,
            epoch_metrics=epoch_metrics,
        )
        for k, v in outputs["logs"].items():
            epoch_metrics[k] = v.float().mean()
        for k, v in epoch_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        epoch_metrics["epoch"] = epoch
        if metadata is not None:
            # Metadata is a dict
            for k, v in metadata.items():
                epoch_metrics[k] = v
        if get_is_master() and not self.args.main.disable_wandb:
            wandb.log(epoch_metrics, step=self.global_step)

        self.args.main.status = f"done {split} epoch"
        return epoch_metrics

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        device,
        epoch,
        loss_scaler,
        lr_scheduler,
        args,
    ):
        raise NotImplementedError

    def evaluate(self, model, dataloader, device, epoch=None, split="val", gather_predictions=False):
        raise NotImplementedError

    def get_required_metric_keys(self, split: Split):
        """Get the required keys for the current split's metrics.

        Returns:
            tuple: (target_keys, pred_keys) where each is a set of required keys
        """
        target_keys = set()
        pred_keys = set()

        # Get keys from epoch metrics
        epoch_metrics = self.get_epoch_metrics(split)
        for metric in epoch_metrics:
            metric_keys = metric.metric_keys
            if "target_label" in metric_keys:
                if isinstance(metric_keys["target_label"], list):
                    target_keys.update(metric_keys["target_label"])
                else:
                    target_keys.add(metric_keys["target_label"])
            if "pred_label" in metric_keys:
                if isinstance(metric_keys["pred_label"], list):
                    pred_keys.update(metric_keys["pred_label"])
                else:
                    pred_keys.add(metric_keys["pred_label"])

        return target_keys, pred_keys
