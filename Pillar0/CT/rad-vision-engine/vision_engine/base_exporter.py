"""Base exporter class with common functionality."""

import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .core.data_structures import ProcessedSeries, ExportResult

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base class for all exporters with common functionality"""

    def __init__(self, config):
        self.config = config

        # Get exporter configuration
        self.exporter_config = config.exporter

        # Get compression type
        self.compression = self.exporter_config.get("compression", "lz4")

        # Get parallel workers setting
        self.workers = config.parallel.get("workers", 4)

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def export(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        """Export a single processed series - must be implemented by subclasses"""
        pass

    def export_tarballs(
        self, processed_series_list: List[ProcessedSeries], output_dir: str
    ) -> Dict[str, Any]:
        """Export multiple processed series as tarballs"""
        raise NotImplementedError("Batch export not implemented for this exporter")

    def _ensure_output_dir(self, output_dir: Path) -> Path:
        """Ensure output directory exists"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_output_filename(self, series_info) -> str:
        """Generate output filename based on compression type"""
        base_name = f"{series_info.accession}.{series_info.series_number}.0"
        if self.compression in ["video", "hevc"]:
            return f"{base_name}.tar"
        else:  # lz4
            return f"{base_name}.tar.lz4"

    def _calculate_dicom_size_mb(self, series_info) -> float:
        """Calculate original DICOM size in MB"""
        dicom_path = series_info.get_dicom_path()
        total_size = 0

        try:
            if os.path.exists(dicom_path):
                for root, _, files in os.walk(dicom_path):
                    for file in files:
                        if file.endswith(".dcm"):
                            total_size += os.path.getsize(os.path.join(root, file))

            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.debug(f"Could not calculate DICOM size: {e}")
            return 0.0
