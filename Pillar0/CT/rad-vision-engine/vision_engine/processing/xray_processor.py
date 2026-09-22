"""
X-ray processor for LZ4/numpy export.
Handles 2D X-ray images preserving raw pixel values.
"""

import os
import logging
from typing import List
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
from skimage import exposure

from ..core.data_structures import SeriesInfo, ProcessedSeries
from ..core.exceptions import ProcessingError
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class XRayProcessor(BaseProcessor):
    """X-ray processor that preserves raw pixel values"""

    def __init__(self, config):
        super().__init__(config)

    def process_series(self, series_info: SeriesInfo) -> ProcessedSeries:
        """Process X-ray series (typically single 2D image)"""
        logger.info(f"Processing X-ray series: {series_info}")

        try:
            # Get DICOM files
            dicom_path = series_info.get_dicom_path()
            if not os.path.exists(dicom_path):
                raise ProcessingError(f"DICOM directory not found: {dicom_path}")

            # Process each DICOM file (X-rays are typically single images)
            numpy_slices = []
            images_shapes_info = []
            dcm_files = [f for f in os.listdir(dicom_path) if f.endswith(".dcm")]

            if not dcm_files:
                raise ProcessingError(f"No DICOM files found in: {dicom_path}")

            for dcm_file in dcm_files:
                dcm_path = os.path.join(dicom_path, dcm_file)
                logger.info(f"Processing DICOM file: {dcm_file}")
                processed_image, shape_info = self._process_xray_image(dcm_path)
                logger.info(
                    f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}, range: [{processed_image.min()}, {processed_image.max()}]"
                )
                numpy_slices.append(processed_image)
                images_shapes_info.append(shape_info)

            # Create processing metadata
            windowing_config = self.config.processing.get("windowing", {})
            windowing_method = windowing_config.get("method", "percentile")

            # Determine processing method name
            if windowing_method == "histogram":
                method_name = "HistogramEqualization"
            elif windowing_method == "percentile":
                method_name = "PercentileWindowing"
            else:
                method_name = "RawValues"

            processing_metadata = {
                "modality": "XR",
                "processing_method": method_name,
                "total_images": len(numpy_slices),
                "value_range": "uint16_raw"
                if method_name == "RawValues"
                else "uint16_normalized",
                "windowing_parameters": windowing_config,
                "per_image_shapes": images_shapes_info,
            }

            return ProcessedSeries(
                series_info=series_info,
                numpy_slices=numpy_slices,
                processing_metadata=processing_metadata,
            )

        except Exception as e:
            raise ProcessingError(
                f"Failed to process X-ray series {series_info.accession}.{series_info.series_number}: {e}"
            )

    def _process_xray_image(self, dcm_path: str):
        """Process single X-ray DICOM image preserving raw pixel values"""
        logger.debug(f"Processing X-ray image: {dcm_path}")

        # Read DICOM
        dcm = pydicom.dcmread(dcm_path)

        # Get pixel array (preserve original dtype)
        image = dcm.pixel_array
        original_shape_hw = tuple(image.shape)
        logger.debug(
            f"Loaded DICOM: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]"
        )

        # Handle photometric interpretation
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                # Invert if needed (MONOCHROME1 = inverted)
                image = np.max(image) - image
                logger.debug(
                    f"Inverted MONOCHROME1: range=[{image.min()}, {image.max()}]"
                )

        # Apply VOI LUT transformation if present
        image = self._apply_voi_lut_if_present(image, dcm)
        logger.debug(
            f"After VOI LUT: shape={image.shape}, range=[{image.min()}, {image.max()}]"
        )

        # Apply windowing based on config
        windowing_config = self.config.processing.get("windowing", {})
        windowing_method = windowing_config.get("method", "percentile")

        if windowing_method == "histogram":
            # Apply histogram equalization (skip percentile windowing entirely)
            logger.debug("Using histogram equalization - skipping percentile windowing")
            # For histogram eq, we'll apply it after resize to maintain full data range
            # Store flag to apply histogram eq later
            apply_histogram_eq = True
        elif windowing_method == "percentile":
            # Apply percentile windowing
            min_percentile = windowing_config.get("min_percentile", 5)
            max_percentile = windowing_config.get("max_percentile", 95)
            image = self._apply_percentile_windowing(
                image, min_percentile, max_percentile
            )
            logger.debug(
                f"After percentile windowing: shape={image.shape}, range=[{image.min()}, {image.max()}]"
            )
            apply_histogram_eq = False
        else:
            # Keep raw values
            apply_histogram_eq = False

        # Resize if requested
        resize = self.config.processing.get("resize", None)
        if resize:
            image = self._resize_image(image, resize)
            logger.debug(
                f"After resize: shape={image.shape}, range=[{image.min()}, {image.max()}]"
            )

        # Apply histogram equalization after resize if requested
        if "apply_histogram_eq" in locals() and apply_histogram_eq:
            image = self._apply_histogram_equalization(image)
            logger.debug(
                f"After histogram eq: shape={image.shape}, range=[{image.min()}, {image.max()}]"
            )

        final_shape_hw = tuple(image.shape)
        logger.debug(
            f"Processed X-ray image shape: {final_shape_hw}, range: [{image.min():.1f}, {image.max():.1f}]"
        )
        shape_info = {
            "source": os.path.basename(dcm_path),
            "original_shape": list(original_shape_hw),
            "final_shape": list(final_shape_hw),
        }
        return image, shape_info

    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to enhance contrast"""
        logger.debug("Applying histogram equalization")

        # Store original dtype for later
        original_dtype = image.dtype
        min_val = image.min()
        max_val = image.max()

        if max_val > min_val:
            # Normalize to 0-1 range for equalization
            normalized = (image.astype(np.float32) - min_val) / (max_val - min_val)

            # Apply histogram equalization using skimage (works on normalized data)
            equalized = exposure.equalize_hist(normalized)

            # Convert back to appropriate bit depth
            if original_dtype == np.uint16 or max_val > 255:
                # Scale to 16-bit range
                result = (equalized * 65535).astype(np.uint16)
            else:
                # Scale to 8-bit range
                result = (equalized * 255).astype(np.uint8)
        else:
            # Handle uniform image
            result = image

        logger.debug(
            f"Histogram equalization complete: range [{result.min()}, {result.max()}]"
        )
        return result

    def _apply_percentile_windowing(
        self, image: np.ndarray, min_percentile: float, max_percentile: float
    ) -> np.ndarray:
        """Apply percentile-based windowing"""
        logger.debug(
            f"Applying percentile windowing: {min_percentile}%-{max_percentile}%"
        )

        # Calculate percentile values
        p_low = np.percentile(image, min_percentile)
        p_high = np.percentile(image, max_percentile)

        if p_high > p_low:
            # Clip and scale to full range
            clipped = np.clip(image, p_low, p_high)

            # Scale to appropriate range based on data type
            if image.dtype == np.uint16 or image.max() > 255:
                # Scale to 16-bit range
                result = ((clipped - p_low) / (p_high - p_low) * 65535).astype(
                    np.uint16
                )
            else:
                # Scale to 8-bit range
                result = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        else:
            # Handle edge case
            result = image

        logger.debug(
            f"Percentile windowing complete: range [{result.min()}, {result.max()}]"
        )
        return result
