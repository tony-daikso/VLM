"""
HEVC Image exporter for Vision Engine - fast 10-bit intra encoding per image.
Uses system ffmpeg with libx265 (Main10) to encode each image as a single-frame MP4.
"""

import os
import io
import json
import logging
import tarfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .core.data_structures import SeriesInfo, ProcessedSeries, ExportResult
from .core.exceptions import ExportError
from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)


@dataclass
class HEVCImageConfig:
    """Configuration for HEVC image compression"""

    crf: int  # Quality (lower is better, typical 12-24)
    preset: str  # x265 preset: ultrafast..placebo
    pix_fmt: str  # Pixel format, e.g., yuv420p10le or yuv444p10le
    container: str  # 'mp4' or 'mkv'


class HEVCImageExporter(BaseExporter):
    """Export 2D series (Mammography, X-ray) as individual HEVC 10-bit files."""

    def __init__(self, config):
        super().__init__(config)
        hevc_cfg = self.exporter_config.get("hevc_image", {})
        self.hevc_cfg = HEVCImageConfig(
            crf=int(hevc_cfg.get("crf", 18)),
            preset=str(hevc_cfg.get("preset", "medium")),
            pix_fmt=str(hevc_cfg.get("pix_fmt", "yuv420p10le")),
            container=str(hevc_cfg.get("container", "mp4")),
        )
        logger.info(
            f"Initialized HEVC image exporter (crf={self.hevc_cfg.crf}, preset={self.hevc_cfg.preset}, "
            f"pix_fmt={self.hevc_cfg.pix_fmt}, container={self.hevc_cfg.container})"
        )

    def export(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        series_info = processed_series.series_info
        # Support any 2D slice modality (e.g., MG, XR)
        if processed_series.numpy_slices is None:
            raise ExportError(
                "HEVCImageExporter supports only 2D slice exports (e.g., MG, XR)"
            )

        # Create a subdirectory named meaningfully (from series description or source directory)
        # Keep the ".0.tar" suffix for consistency with legacy naming
        folder_base = self._derive_series_basename(series_info)
        tar_style_name = f"{folder_base}.0.tar"
        output_dir = Path(output_dir) / tar_style_name
        output_dir.mkdir(parents=True, exist_ok=True)

        total_size_mb = 0.0
        count = 0

        try:
            for i, slice_array in enumerate(processed_series.numpy_slices):
                # Build output filename from best available series name
                base = self._derive_series_basename(series_info)
                name = (
                    base
                    if len(processed_series.numpy_slices) == 1
                    else f"{base}_{i:04d}"
                )
                out_path = output_dir / f"{name}.{self.hevc_cfg.container}"

                if out_path.exists() and not getattr(self.config, "overwrite", False):
                    logger.info(f"Skipping existing file: {out_path}")
                    total_size_mb += out_path.stat().st_size / (1024 * 1024)
                    count += 1
                    continue

                # Write a temporary 16-bit PNG as input to ffmpeg
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_png = Path(tmpdir) / "frame.png"
                    png_bytes = self._array_to_png16(slice_array)
                    with open(tmp_png, "wb") as f:
                        f.write(png_bytes)

                    # ffmpeg command: single frame, intra-only, main10
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(tmp_png),
                        "-c:v",
                        "libx265",
                        "-pix_fmt",
                        self.hevc_cfg.pix_fmt,
                        "-x265-params",
                        "keyint=1:scenecut=0:open-gop=0:repeat-headers=1",
                        "-preset",
                        self.hevc_cfg.preset,
                        "-crf",
                        str(self.hevc_cfg.crf),
                        "-frames:v",
                        "1",
                        str(out_path),
                    ]
                    logger.debug(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode != 0:
                        raise ExportError(
                            f"ffmpeg failed: {result.stderr.decode('utf-8', 'ignore')}"
                        )

                total_size_mb += out_path.stat().st_size / (1024 * 1024)
                count += 1

            # Save metadata JSON next to outputs
            meta_path = (
                output_dir
                / f"{series_info.accession}_{series_info.series_number}_metadata.json"
            )
            with open(meta_path, "w") as f:
                f.write(self._create_metadata_json(processed_series))

            return ExportResult(
                series_info=series_info,
                output_path=str(output_dir),
                file_size_mb=total_size_mb,
                slice_count=count,
                success=True,
            )
        except Exception as e:
            raise ExportError(f"Failed to export HEVC images: {e}")

    def _array_to_png16(self, array: np.ndarray) -> bytes:
        # Ensure 16-bit PNG input for ffmpeg
        if array.dtype == np.uint16:
            img = Image.fromarray(array, mode="I;16")
        elif array.dtype == np.int16:
            arr = (array.astype(np.int32) + 32768).astype(np.uint16)
            img = Image.fromarray(arr, mode="I;16")
        else:
            # Normalize to 16-bit
            min_val = float(np.min(array))
            max_val = float(np.max(array))
            if max_val > min_val:
                norm = ((array - min_val) / (max_val - min_val) * 65535.0).astype(
                    np.uint16
                )
            else:
                norm = np.zeros_like(array, dtype=np.uint16)
            img = Image.fromarray(norm, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=3)
        return buf.getvalue()

    def _create_metadata_json(self, processed_series: ProcessedSeries) -> str:
        metadata = {
            "series_info": {
                "accession": processed_series.series_info.accession,
                "series_number": processed_series.series_info.series_number,
                "series_uid": processed_series.series_info.series_uid,
                "series_description": processed_series.series_info.series_description,
                "modality": processed_series.series_info.modality,
                "slice_count": processed_series.get_slice_count(),
            },
            "processing_metadata": processed_series.processing_metadata,
            "export_info": {
                "format": "HEVC",
                "profile": "Main10",
                "pix_fmt": self.hevc_cfg.pix_fmt,
                "crf": self.hevc_cfg.crf,
                "preset": self.hevc_cfg.preset,
                "container": self.hevc_cfg.container,
                "exporter_version": "1.0",
            },
        }
        return json.dumps(metadata, indent=2)

    def _derive_series_basename(self, series_info: SeriesInfo) -> str:
        """Derive a human-meaningful base filename for outputs.
        Preference: series_description -> directory/file name -> series_uid -> accession/series_number.
        Avoids generic 'unknown_*' names."""
        # 1) Series description
        desc = (getattr(series_info, "series_description", "") or "").strip()
        if desc:
            return self._sanitize_name(desc)
        # 2) Directory or file name from path
        try:
            path = series_info.get_dicom_path()
            base = os.path.basename(path.rstrip("/"))
            if os.path.isfile(path):
                base = os.path.splitext(base)[0]
            if base:
                return self._sanitize_name(base)
        except Exception:
            pass
        # 3) Series UID
        uid = (getattr(series_info, "series_uid", "") or "").strip()
        if uid:
            return self._sanitize_name(uid)
        # 4) Accession + series number (last resort)
        acc = (getattr(series_info, "accession", "") or "").strip()
        if acc and not acc.startswith("unknown_"):
            return self._sanitize_name(f"{acc}_{series_info.series_number}")
        return self._sanitize_name(f"series_{series_info.series_number}")

    def _sanitize_name(self, name: str) -> str:
        allowed = []
        for ch in name:
            if ch.isalnum() or ch in ["-", "_", "."]:
                allowed.append(ch)
            else:
                allowed.append("_")
        # Collapse repeated underscores
        sanitized = "".join(allowed)
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_") or "series"
