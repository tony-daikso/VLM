"""Loss functions."""

from pillar.losses.survival import SurvivalLoss
from pillar.losses.object_prediction import DETRObjectDetectionLoss, SybilRegionAnnotationLoss

__all__ = ["SurvivalLoss", "DETRObjectDetectionLoss", "SybilRegionAnnotationLoss"]
