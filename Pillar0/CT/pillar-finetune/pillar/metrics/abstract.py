from abc import ABC, abstractmethod

from torch import Tensor


class AbstractMetric(ABC):
    def __init__(self, args, **kwargs):
        self.args = args

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, Tensor]:
        pass

    @property
    def metric_keys(self) -> dict[str, str] | dict[str, list[str]]:
        """
        Returns a dictionary with the keys of the target (from batch) and the keys of the prediction (from model_output)
        e.g. {'target_label': 'y', 'pred_label': 'logit'}
        """
        return {}
