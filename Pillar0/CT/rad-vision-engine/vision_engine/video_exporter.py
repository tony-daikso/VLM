"""
Video codec exporter using HEVC for CT volume compression.
Supports both CPU (libx265) and GPU (hevc_nvenc) encoding.

Treats CT slices as video frames and uses modern video codecs
for efficient compression while preserving diagnostic quality.
"""

import io
import os
import json
import tarfile
import tempfile
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .core.data_structures import ProcessedSeries, ExportResult
from .core.exceptions import ExportError
from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video encoding parameters."""

    codec: str  # Video codec to use
    bit_depth: int  # Bit depth for encoding
    crf: int  # Constant Rate Factor (lower = higher quality)
    gop_size: int  # Group of Pictures size
    hu_min: int  # Minimum HU value to map
    hu_max: int  # Maximum HU value to map
    preset: str  # Encoding preset (slower = better compression)
    gpu_id: Optional[int] = None  # GPU device ID for NVENC
    archive: bool = True  # Whether to archive output into a tarball
    lossless: bool = False  # Enable x265 lossless mode (CPU encoder only)


class HardwareEncodingSupport:
    """Check and manage hardware encoding capabilities."""

    @staticmethod
    def get_encoder_and_params(video_config: VideoConfig) -> Tuple[str, list]:
        """CPU-only: always use libx265 with requested preset/CRF (no NVENC checks)."""
        logger.info("Using CPU encoding (libx265)")
        encoder = "libx265"
        x265_params = (
            f"keyint={video_config.gop_size}:min-keyint={video_config.gop_size}"
        )
        if getattr(video_config, "lossless", False):
            logger.info("x265 lossless mode enabled")
            codec_params = [
                "-c:v",
                encoder,
                "-preset",
                video_config.preset,
                "-pix_fmt",
                "yuv420p10le",
                "-x265-params",
                x265_params + ":lossless=1",
            ]
        else:
            codec_params = [
                "-c:v",
                encoder,
                "-preset",
                video_config.preset,
                "-crf",
                str(video_config.crf),
                "-pix_fmt",
                "yuv420p10le",
                "-x265-params",
                x265_params,
            ]
        return encoder, codec_params


class VideoExporter(BaseExporter):
    """Export CT volumes as video files using HEVC compression with GPU support.

    Supports two output modes controlled by config:
    - Archived (default): write a .tar containing volume.mkv and metadata.json
    - Unarchived: write a folder with volume.mkv and metadata.json
    """

    def __init__(self, config):
        super().__init__(config)

        # Get video-specific configuration
        video_config = self.exporter_config.get("video", {})

        # Video encoding settings
        self.video_config = VideoConfig(
            codec=video_config["codec"],
            bit_depth=video_config["bit_depth"],
            crf=video_config["crf"],
            gop_size=video_config["gop_size"],
            hu_min=video_config["hu_min"],
            hu_max=video_config["hu_max"],
            preset=video_config["preset"],
            gpu_id=video_config.get("gpu_id", None),
            archive=video_config.get("archive", True),
        )

        # Validate codec availability
        self._check_codec_availability()

        # Video is already highly compressed, gzip adds minimal benefit
        self.use_gzip = False

        # Get encoder info for logging
        encoder, _ = HardwareEncodingSupport.get_encoder_and_params(self.video_config)
        logger.info(f"Initialized video exporter with {encoder} encoder")
        logger.info(
            f"HU range: [{self.video_config.hu_min}, {self.video_config.hu_max}]"
        )
        logger.info(
            f"CRF/CQ: {self.video_config.crf}, GOP: {self.video_config.gop_size}"
        )
        if self.video_config.gpu_id is not None:
            logger.info(f"Using GPU: {self.video_config.gpu_id}")

    def _check_codec_availability(self):
        """Check if required video codec is available."""
        try:
            # Check ffmpeg availability
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise ExportError("ffmpeg is not available. Please install ffmpeg.")

            # Get encoder info
            encoder, _ = HardwareEncodingSupport.get_encoder_and_params(
                self.video_config
            )

            # Check for specific encoder support
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True
            )
            if encoder not in result.stdout:
                raise ExportError(f"{encoder} encoder not available in ffmpeg")

        except FileNotFoundError:
            raise ExportError(
                "ffmpeg not found. Please install ffmpeg for video encoding."
            )

    def export(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        """Export processed series as a video deliverable (tarball or folder)."""
        series_info = processed_series.series_info
        base_name = f"{series_info.accession}.{series_info.series_number}.0"
        # Determine desired video container/extension from exporter.output.extension (default to .mkv)
        try:
            desired_ext = self.config.exporter.get("output", {}).get(
                "extension", ".mkv"
            )
        except Exception:
            desired_ext = ".mkv"
        if not str(desired_ext).startswith("."):
            desired_ext = f".{desired_ext}"
        is_archived = self.video_config.archive
        output_path = output_dir / (f"{base_name}.tar" if is_archived else base_name)

        # Check if target exists and overwrite is not set
        if is_archived:
            if output_path.exists() and not getattr(self.config, "overwrite", False):
                logger.info(f"Skipping existing file: {output_path}")
                return ExportResult(
                    series_info=series_info,
                    output_path=str(output_path),
                    file_size_mb=output_path.stat().st_size / (1024 * 1024),
                    slice_count=0,
                    success=True,
                )
        else:
            dest_dir = output_path
            dest_video_path = dest_dir / "volume.mkv"
            dest_metadata_path = dest_dir / "metadata.json"
            if (
                dest_video_path.exists()
                and dest_metadata_path.exists()
                and not getattr(self.config, "overwrite", False)
            ):
                logger.info(f"Skipping existing folder: {dest_dir}")
                # Report size as video size
                file_size_mb = dest_video_path.stat().st_size / (1024 * 1024)
                return ExportResult(
                    series_info=series_info,
                    output_path=str(dest_dir),
                    file_size_mb=file_size_mb,
                    slice_count=0,
                    success=True,
                )

        try:
            # Get the numpy volume
            if processed_series.numpy_volume is not None:
                volume = processed_series.numpy_volume
            else:
                # Stack slices if we have them
                volume = np.stack(processed_series.numpy_slices)

            # Encode to video
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                video_path = temp_path / f"volume{desired_ext}"

                # Encode volume
                self._encode_volume_as_video(volume, video_path)

                # Create metadata
                metadata = self._create_metadata_json(processed_series)

                if is_archived:
                    # Create tarball with video and metadata
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with tarfile.open(output_path, mode="w") as tar:
                        # Add video file
                        tar.add(video_path, arcname=f"volume{desired_ext}")

                        # Add metadata
                        metadata_info = tarfile.TarInfo(name="metadata.json")
                        metadata_bytes = metadata.encode("utf-8")
                        metadata_info.size = len(metadata_bytes)
                        tar.addfile(metadata_info, io.BytesIO(metadata_bytes))

                    return ExportResult(
                        series_info=series_info,
                        output_path=str(output_path),
                        file_size_mb=output_path.stat().st_size / (1024 * 1024),
                        slice_count=len(volume),
                        success=True,
                    )
                else:
                    # Write to folder: volume.mkv and metadata.json
                    dest_dir = output_path
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_video_path = dest_dir / f"volume{desired_ext}"
                    dest_metadata_path = dest_dir / "metadata.json"

                    # Move video atomically
                    try:
                        shutil.move(str(video_path), str(dest_video_path))
                    except Exception as move_err:
                        raise ExportError(
                            f"Failed to move encoded video to destination: {move_err}"
                        )

                    # Write metadata
                    try:
                        with open(dest_metadata_path, "w", encoding="utf-8") as f:
                            f.write(metadata)
                    except Exception as write_err:
                        raise ExportError(f"Failed to write metadata.json: {write_err}")

                    return ExportResult(
                        series_info=series_info,
                        output_path=str(dest_dir),
                        file_size_mb=dest_video_path.stat().st_size / (1024 * 1024),
                        slice_count=len(volume),
                        success=True,
                    )

        except Exception as e:
            raise ExportError(f"Failed to export series {series_info}: {e}")

    def _encode_volume_as_video(self, volume: np.ndarray, output_path: Path):
        """Encode medical volume as video file using ffmpeg with GPU support."""
        # Handle different modalities - CT uses int16, MR uses uint16
        if self.video_config.hu_min < 0:
            # CT mode with negative HU values
            if volume.dtype != np.int16:
                logger.warning(f"Converting volume from {volume.dtype} to int16 for CT")
                volume = volume.astype(np.int16)
        else:
            # MR/other modalities with positive values only
            if volume.dtype != np.uint16:
                logger.warning(
                    f"Converting volume from {volume.dtype} to uint16 for MR/other"
                )
                volume = volume.astype(np.uint16)

        # Get dimensions
        num_slices, height, width = volume.shape

        # Map HU values to 10-bit range (0-1023)
        hu_range = self.video_config.hu_max - self.video_config.hu_min

        # Debug: Check input data
        logger.info(
            f"Input volume stats - Min: {volume.min()}, Max: {volume.max()}, Mean: {volume.mean():.1f}, Std: {volume.std():.1f}"
        )

        # Clip and map to 10-bit
        volume_clipped = np.clip(
            volume, self.video_config.hu_min, self.video_config.hu_max
        )
        # Scale to full 16-bit range for ffmpeg input (will be converted to 10-bit by encoder)
        # FFmpeg expects 16-bit input to use full range, not just 0-1023
        volume_16bit = (
            (volume_clipped - self.video_config.hu_min) * 65535.0 / hu_range
        ).astype(np.uint16)

        # Debug: Check output data
        logger.info(
            f"16-bit volume stats - Min: {volume_16bit.min()}, Max: {volume_16bit.max()}, Mean: {volume_16bit.mean():.1f}, Std: {volume_16bit.std():.1f}"
        )

        # Get encoder and parameters
        encoder, codec_params = HardwareEncodingSupport.get_encoder_and_params(
            self.video_config
        )

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f",
            "rawvideo",
            "-video_size",
            f"{width}x{height}",
            "-pixel_format",
            "gray16le",  # Input is 16-bit grayscale
            "-framerate",
            "25",  # Arbitrary framerate
            "-i",
            "-",  # Read from stdin
            *codec_params,
            "-color_range",
            "pc",  # Full/PC range (0-1023)
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            str(output_path),
        ]

        logger.info(f"Encoding {num_slices} slices as HEVC video using {encoder}...")

        # Prepare all frame data upfront
        all_frames = []
        for i in range(num_slices):
            frame = volume_16bit[i]
            # Ensure C-contiguous array for writing
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            all_frames.append(frame.tobytes())

        input_data = b"".join(all_frames)

        # Run ffmpeg with all data at once
        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise ExportError(
                    f"ffmpeg encoding failed (code {result.returncode}): {result.stderr.decode()}"
                )

            # Check if output file was created
            if not output_path.exists():
                raise ExportError(f"ffmpeg did not create output file: {output_path}")

            logger.info(
                f"Video encoding complete: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )

        except subprocess.TimeoutExpired:
            raise ExportError("ffmpeg encoding timed out after 5 minutes")
        except Exception as e:
            raise ExportError(
                f"Error during video encoding: {type(e).__name__}: {str(e)}"
            )

    def _create_metadata_json(self, processed_series: ProcessedSeries) -> str:
        """Create metadata JSON for the series."""
        encoder, _ = HardwareEncodingSupport.get_encoder_and_params(self.video_config)

        metadata = {
            "series_info": {
                "accession": processed_series.series_info.accession,
                "series_number": processed_series.series_info.series_number,
                "series_uid": processed_series.series_info.series_uid,
                "series_description": processed_series.series_info.series_description,
                "modality": processed_series.series_info.modality,
                "slice_count": processed_series.series_info.slice_count,
            },
            "processing_metadata": processed_series.processing_metadata,
            "export_info": {
                "format": "video",
                "encoder": encoder,
                "codec": self.video_config.codec.upper(),
                "bit_depth": self.video_config.bit_depth,
                "crf": self.video_config.crf,
                "gop_size": self.video_config.gop_size,
                "archived": self.video_config.archive,
                "hu_mapping": {
                    "min": self.video_config.hu_min,
                    "max": self.video_config.hu_max,
                    "output_range": [0, 1023],
                },
                "exporter_version": "1.1",
                "hardware_accelerated": "nvenc" in encoder,
            },
            "source_data": {
                "dicom_directory": processed_series.series_info.get_dicom_path(),
                "study_directory": processed_series.series_info.study_path,
            },
        }

        return json.dumps(metadata, indent=2)
