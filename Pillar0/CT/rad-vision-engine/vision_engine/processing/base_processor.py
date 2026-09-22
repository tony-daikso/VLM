"""Base processor with common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ..core.data_structures import SeriesInfo, ProcessedSeries
from ..core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Base class for medical image processors"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get processing config
        self.processing_config = config.processing

    @abstractmethod
    def process_series(self, series_info: SeriesInfo) -> ProcessedSeries:
        """Process a series - must be implemented by subclasses"""
        pass

    # Removed apply_windowing methods - no longer needed with LZ4/numpy storage

    def _apply_voi_lut_if_present(self, image: np.ndarray, dcm) -> np.ndarray:
        """
        Apply VOI LUT transformation if present in DICOM.

        Args:
            image: Input image array
            dcm: DICOM dataset object

        Returns:
            Image with VOI LUT applied (if available), otherwise original image
        """
        try:
            # Apply VOI LUT using pydicom's built-in function
            # This respects WindowCenter/WindowWidth or actual LUT data in DICOM
            image_voi = apply_voi_lut(
                image, dcm, index=0
            )  # Use first VOI LUT if multiple exist
            logger.debug("Applied VOI LUT transformation")
            return image_voi  # Keep original dtype
        except (AttributeError, ValueError, KeyError, IndexError) as e:
            # VOI LUT not available or not applicable - continue without it
            logger.debug(f"VOI LUT not applied: {e}")
            return image

    def _resize_image(
        self, image: np.ndarray, target_size: Optional[List[int]]
    ) -> np.ndarray:
        """
        Resize a 2D image while preserving dtype and value range.

        Args:
            image: 2D numpy array to resize
            target_size: Target size as [height, width]. If None, returns original image.

        Returns:
            Resized image with original dtype and value range preserved
        """
        if target_size is None:
            return image

        min_val = image.min()
        max_val = image.max()

        if max_val > min_val:
            # Normalize to float32 for OpenCV
            normalized = ((image - min_val) / (max_val - min_val)).astype(np.float32)
            # OpenCV resize - note cv2.resize expects (width, height)
            resized_norm = cv2.resize(
                normalized,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            # Convert back to original dtype and restore original range
            resized = (resized_norm * (max_val - min_val) + min_val).astype(image.dtype)
        else:
            # Handle edge case of uniform image
            resized = np.full(target_size, min_val, dtype=image.dtype)

        return resized
