from typing import Tuple
from abc import ABC, abstractmethod

from torch import Tensor


class AbstractLoss(ABC):
    def __init__(self, args, **kwargs):
        self.args = args

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        pass

    @property
    def loss_keys(self) -> dict[str, str] | dict[str, list[str]]:
        """
        Returns a dictionary with the keys of the target (from batch) and the keys of the prediction (from model_output)
        e.g. {'target_label': 'y', 'pred_label': 'logit'}
        """
        return {}
