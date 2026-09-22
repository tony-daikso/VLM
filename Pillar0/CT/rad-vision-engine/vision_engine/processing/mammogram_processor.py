"""
Mammogram processor for LZ4/numpy export.
Handles 2D mammography images preserving raw pixel values.
"""

import os
import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from ..core.data_structures import SeriesInfo, ProcessedSeries
from ..core.exceptions import ProcessingError
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class MammogramProcessor(BaseProcessor):
    """Mammogram processor that preserves raw pixel values"""

    def __init__(self, config):
        super().__init__(config)
        # Track last resampling/crop results for metadata
        self._last_original_spacing: Optional[Tuple[float, float]] = (
            None  # (x, y) in mm
        )
        self._last_target_spacing: Optional[Tuple[float, float]] = None  # (x, y) in mm
        self._last_original_shape: Optional[Tuple[int, int]] = (
            None  # (H, W) before any processing
        )
        self._last_final_shape: Optional[Tuple[int, int]] = (
            None  # (H, W) after all processing
        )

    def process_series(self, series_info: SeriesInfo) -> ProcessedSeries:
        """Process mammogram series (typically single 2D image)"""
        logger.info(f"Processing mammogram series: {series_info}")

        try:
            # Get DICOM files
            dicom_path = series_info.get_dicom_path()
            if not os.path.exists(dicom_path):
                raise ProcessingError(f"DICOM directory not found: {dicom_path}")

            # Process each DICOM file (mammograms are typically single images)
            numpy_slices = []
            images_shapes_info = []
            dcm_files = [f for f in os.listdir(dicom_path) if f.endswith(".dcm")]

            if not dcm_files:
                raise ProcessingError(f"No DICOM files found in: {dicom_path}")

            for dcm_file in dcm_files:
                dcm_path = os.path.join(dicom_path, dcm_file)
                try:
                    processed_image, shape_info = self._process_mammogram_image(
                        dcm_path
                    )
                    if processed_image is not None:
                        numpy_slices.append(processed_image)
                        images_shapes_info.append(shape_info)
                except Exception as e:
                    # Skip files that can't be processed (e.g., SR files without pixel data)
                    logger.debug(f"Skipping file {dcm_file}: {e}")
                    continue

            # Check if we have any valid images
            if not numpy_slices:
                raise ProcessingError(
                    f"No valid mammography images found in series {series_info.accession}.{series_info.series_number}"
                )

            # Create processing metadata
            windowing_config = self.config.processing.get("windowing", {})
            windowing_method = windowing_config.get("method", "none")

            # Determine processing method and value range
            if windowing_method == "minmax":
                method_name = "MinMaxNormalization"
                value_range = "uint16_normalized"
            elif windowing_method == "percentile":
                method_name = "PercentileWindowing"
                value_range = "uint16_normalized"
            else:
                method_name = "RawValues"
                value_range = "raw_pixel_values"

            # Resampling and crop/pad targets
            resampling_config = self.config.processing.get("resampling", {})
            target_spacing = resampling_config.get("target_spacing", None)
            crop_pad_config = self.config.processing.get("crop_pad", {})
            crop_pad_target = crop_pad_config.get("size", None)

            # Determine final shape from first processed image if available
            final_shape = tuple(numpy_slices[0].shape) if numpy_slices else None
            self._last_final_shape = final_shape

            processing_metadata = {
                "modality": "MG",
                "processing_method": method_name,
                "total_images": len(numpy_slices),
                "value_range": value_range,
                "windowing_parameters": windowing_config,
                "original_shape": list(self._last_original_shape)
                if self._last_original_shape
                else None,
                "original_spacing": list(self._last_original_spacing)
                if self._last_original_spacing
                else None,
                "target_spacing": target_spacing,
                "final_shape": list(final_shape) if final_shape else None,
                "crop_pad_target": crop_pad_target,
                "series_instance_uid": series_info.series_uid,
                "per_image_shapes": images_shapes_info,
            }

            return ProcessedSeries(
                series_info=series_info,
                numpy_slices=numpy_slices,
                processing_metadata=processing_metadata,
            )

        except Exception as e:
            raise ProcessingError(
                f"Failed to process mammogram series {series_info.accession}.{series_info.series_number}: {e}"
            )

    def _process_mammogram_image(self, dcm_path: str):
        """Process single mammogram DICOM image preserving raw pixel values"""
        logger.debug(f"Processing mammogram image: {dcm_path}")

        # Read DICOM
        dcm = pydicom.dcmread(dcm_path)

        # Check if this is actually a mammography image
        if hasattr(dcm, "Modality") and dcm.Modality not in ["MG", "DX"]:
            logger.debug(f"Skipping non-mammography file with modality: {dcm.Modality}")
            return None

        # Check if pixel data exists
        if not hasattr(dcm, "PixelData"):
            logger.debug(f"Skipping file without pixel data")
            return None

        # Get pixel array (preserve original dtype)
        image = dcm.pixel_array

        # Track original shape before any processing
        self._last_original_shape = tuple(image.shape)
        original_shape_hw = tuple(image.shape)
        resampled_shape_hw = None
        logger.debug(f"Original mammogram shape: {self._last_original_shape}")

        # Handle photometric interpretation (mammograms can vary)
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                # Invert if needed (MONOCHROME1 = inverted, higher values = darker)
                image = np.max(image) - image

        # Apply VOI LUT transformation if present
        image = self._apply_voi_lut_if_present(image, dcm)

        # Handle Presentation LUT (common in mammography) - after VOI LUT
        if (
            hasattr(dcm, "PresentationLUTShape")
            and dcm.PresentationLUTShape == "INVERSE"
        ):
            image = np.max(image) - image

        # Optional 2D resampling using PixelSpacing and target_spacing from config
        try:
            resampling_config = self.config.processing.get("resampling", {})
            target_spacing = resampling_config.get("target_spacing", None)
            # Extract spacing from multiple possible DICOM tags
            pixel_spacing = self._extract_pixel_spacing(dcm)
            if target_spacing and pixel_spacing:
                # Store original spacing as (x, y)
                original_spacing_xy = (float(pixel_spacing[0]), float(pixel_spacing[1]))
                target_spacing_xy = (float(target_spacing[0]), float(target_spacing[1]))
                logger.info(
                    f"MG 2D resample: spacing {original_spacing_xy} → {target_spacing_xy}"
                )
                image = self._resample_image_2d(
                    image, original_spacing_xy, target_spacing_xy
                )
                resampled_shape_hw = tuple(image.shape)
                self._last_original_spacing = original_spacing_xy
                self._last_target_spacing = target_spacing_xy
            else:
                # Track spacing if available even when not resampling
                if pixel_spacing:
                    self._last_original_spacing = (
                        float(pixel_spacing[0]),
                        float(pixel_spacing[1]),
                    )
                else:
                    logger.debug(
                        "No pixel spacing tags found; metadata will have null spacing"
                    )
        except Exception as e:
            logger.warning(f"2D resampling skipped due to error: {e}")

        # Apply center crop/pad if requested (replaces old resize behavior)
        crop_pad = self.config.processing.get("crop_pad", None)
        if crop_pad and crop_pad.get("size"):
            before_shape = tuple(image.shape)
            image = self._center_crop_pad_2d(image, crop_pad["size"])
            after_shape = tuple(image.shape)
            logger.info(
                f"MG crop/pad: {before_shape} → {after_shape} (target {crop_pad['size']})"
            )

        # Apply windowing based on config
        windowing_config = self.config.processing.get("windowing", {})
        windowing_method = windowing_config.get("method", "none")

        if windowing_method == "minmax":
            # Apply min-max normalization for easier viewing
            logger.debug("Applying min-max normalization")
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                # Scale to full 16-bit range for export
                image = ((image - min_val) / (max_val - min_val) * 65535).astype(
                    np.uint16
                )
            else:
                # Handle edge case of uniform image
                image = np.zeros_like(image, dtype=np.uint16)
            logger.debug(
                f"After min-max normalization: range=[{image.min()}, {image.max()}]"
            )
        elif windowing_method == "percentile":
            # Apply percentile windowing
            min_percentile = windowing_config.get("min_percentile", 5)
            max_percentile = windowing_config.get("max_percentile", 95)
            p_low = np.percentile(image, min_percentile)
            p_high = np.percentile(image, max_percentile)
            if p_high > p_low:
                clipped = np.clip(image, p_low, p_high)
                image = ((clipped - p_low) / (p_high - p_low) * 65535).astype(np.uint16)
            else:
                image = np.zeros_like(image, dtype=np.uint16)
            logger.debug(
                f"After percentile windowing ({min_percentile}-{max_percentile}%): range=[{image.min()}, {image.max()}]"
            )
        else:
            # Keep raw values (method == 'none')
            # Ensure uint16 dtype
            if image.dtype != np.uint16:
                if image.max() <= 65535:
                    image = image.astype(np.uint16)
                else:
                    logger.warning(
                        f"Image values exceed uint16 range ({image.max()}), scaling down"
                    )
                    image = (image / image.max() * 65535).astype(np.uint16)

        final_shape_hw = tuple(image.shape)
        logger.debug(
            f"Processed mammogram image shape: {final_shape_hw}, range: [{image.min():.1f}, {image.max():.1f}]"
        )
        shape_info = {
            "source": os.path.basename(dcm_path),
            "original_shape": list(original_shape_hw),
            "resampled_shape": list(resampled_shape_hw) if resampled_shape_hw else None,
            "final_shape": list(final_shape_hw),
        }
        return image, shape_info

    def _extract_pixel_spacing(self, dcm) -> Optional[Tuple[float, float]]:
        """Extract (x, y) pixel spacing in mm from various DICOM tags.
        Order of precedence: PixelSpacing (0028,0030) -> ImagerPixelSpacing (0018,1164)
        -> SpacingBetweenColumns/Rows (0018,1166/0018,1165). Returns None if not found."""
        try:
            # PixelSpacing: [row_spacing, col_spacing] in mm
            if (
                hasattr(dcm, "PixelSpacing")
                and dcm.PixelSpacing
                and len(dcm.PixelSpacing) >= 2
            ):
                row, col = float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])
                return (col, row)  # return (x, y)
        except Exception:
            pass
        try:
            # ImagerPixelSpacing: [row_spacing, col_spacing]
            if (
                hasattr(dcm, "ImagerPixelSpacing")
                and dcm.ImagerPixelSpacing
                and len(dcm.ImagerPixelSpacing) >= 2
            ):
                row, col = (
                    float(dcm.ImagerPixelSpacing[0]),
                    float(dcm.ImagerPixelSpacing[1]),
                )
                return (col, row)
        except Exception:
            pass
        try:
            # SpacingBetweenRows/Columns (rare)
            row = getattr(dcm, "SpacingBetweenRows", None)
            col = getattr(dcm, "SpacingBetweenColumns", None)
            if row is not None and col is not None:
                return (float(col), float(row))
        except Exception:
            pass
        # Could add functional groups parsing for multiframe here if needed
        return None

    def _resample_image_2d(
        self,
        image: np.ndarray,
        original_spacing_xy: Tuple[float, float],
        target_spacing_xy: Tuple[float, float],
    ) -> np.ndarray:
        """Resample a 2D image to the target (x, y) spacing using SimpleITK (linear)."""
        # Preserve original dtype
        original_dtype = image.dtype
        # SimpleITK expects (width, height) size and (x, y) spacing
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image.SetSpacing(
            (float(original_spacing_xy[0]), float(original_spacing_xy[1]))
        )

        original_size = sitk_image.GetSize()  # (width, height)
        original_spacing = sitk_image.GetSpacing()  # (x, y)
        target_spacing = (float(target_spacing_xy[0]), float(target_spacing_xy[1]))

        # Compute new size in SITK order (width, height)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(2)
        ]
        logger.info(f"MG 2D resample size: {original_size} → {new_size}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        # Use background as the minimum value to minimize artifacts
        resampler.SetDefaultPixelValue(float(np.min(image)))
        resampler.SetInterpolator(sitk.sitkLinear)

        resampled = resampler.Execute(sitk_image)
        resampled_np = sitk.GetArrayFromImage(resampled)

        # Convert back to original dtype
        return resampled_np.astype(original_dtype, copy=False)

    def _center_crop_pad_2d(
        self, image: np.ndarray, target_hw: List[int]
    ) -> np.ndarray:
        """Center crop or pad a 2D image to the target (H, W)."""
        original_dtype = image.dtype
        pad_value = float(image.min())
        tensor = torch.from_numpy(image).float()
        current_h, current_w = tensor.shape[-2], tensor.shape[-1]
        target_h, target_w = int(target_hw[0]), int(target_hw[1])

        h_diff = target_h - current_h
        w_diff = target_w - current_w

        if h_diff < 0:
            h_start = (-h_diff) // 2
            tensor = tensor[h_start : h_start + target_h, :]
        elif h_diff > 0:
            h_pad = (h_diff // 2, h_diff - h_diff // 2)
            tensor = F.pad(tensor, (0, 0, *h_pad), mode="constant", value=pad_value)

        if w_diff < 0:
            w_start = (-w_diff) // 2
            tensor = tensor[:, w_start : w_start + target_w]
        elif w_diff > 0:
            w_pad = (w_diff // 2, w_diff - w_diff // 2)
            tensor = F.pad(tensor, (*w_pad, 0, 0), mode="constant", value=pad_value)

        return tensor.numpy().astype(original_dtype, copy=False)
