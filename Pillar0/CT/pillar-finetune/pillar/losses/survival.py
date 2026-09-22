from collections import OrderedDict

import torch
import torch.nn.functional as F
from pillar.losses.abstract import AbstractLoss


class SurvivalLoss(AbstractLoss):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.logit_key = kwargs.get("logit_key", "logit")
        assert self.args.engine.kwargs.binary_pred, (
            "args.engine.kwargs.binary_pred should be set to True if SurvivalLoss is used"
        )

    def __call__(self, batch=None, model_output=None, **extras):
        logit = model_output[self.logit_key]

        logging_dict = OrderedDict()
        y_seq, y_mask = batch["y_seq"], batch["y_mask"]
        loss = F.binary_cross_entropy_with_logits(
            logit, y_seq.float(), weight=y_mask.float(), reduction="sum"
        ) / torch.sum(y_mask.float())
        logging_dict[f"survival_loss_{self.logit_key}"] = loss.detach()

        return loss, logging_dict

    @property
    def loss_keys(self):
        return {"target_label": ["y_seq", "y_mask"], "pred_label": self.logit_key}
