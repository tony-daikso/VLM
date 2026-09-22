"""
Common data structures for vision-engine pipeline
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np


@dataclass
class SeriesInfo:
    """Information about a DICOM series"""

    accession: str
    series_uid: str
    study_path: str
    series_number: int
    series_description: str
    slice_count: int
    modality: str = "CT"

    # Optional metadata
    slice_thickness: Optional[float] = None
    acquisition_time: Optional[str] = None
    anatomy: Optional[str] = None

    def get_dicom_path(self) -> str:
        """Construct path to DICOM series directory (following reference code logic)"""
        # Check if we have a direct path override (from series_paths_loader)
        if hasattr(self, "_direct_path"):
            return self._direct_path

        # Handle reference code path structure
        study_path = self.study_path.replace(
            "AIR_API_Downloads", "YAIB-cohorts/AIR_API_Downloads"
        )

        # Check if DICOM files are directly in the study path (flat structure)
        # This is common for anonymized datasets
        try:
            files_in_study = os.listdir(study_path)
            dcm_files = [
                f
                for f in files_in_study
                if f.endswith(".dcm")
                or self._is_dicom_file(os.path.join(study_path, f))
            ]
            if dcm_files:
                # DICOM files are directly in study path, return it
                return study_path
        except (FileNotFoundError, PermissionError, OSError):
            pass

        try:
            # Find subdirectory (like reference code)
            subdirs = [
                d
                for d in os.listdir(study_path)
                if "xlsx" not in d and os.path.isdir(os.path.join(study_path, d))
            ]
            if subdirs:
                subfolder = subdirs[0]
                return f"{study_path}/{subfolder}/{self.series_uid}"
            else:
                # Fallback to direct path
                return f"{study_path}/{self.series_uid}"
        except (FileNotFoundError, PermissionError):
            # Fallback if directory doesn't exist or can't be read
            return f"{study_path}/{self.series_uid}"

    def _is_dicom_file(self, filepath: str) -> bool:
        """Quick check if file is DICOM"""
        try:
            with open(filepath, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except:
            return False

    def get_output_filename(self, phase: int = 0) -> str:
        """Generate output filename for tarball"""
        return f"{self.accession}.{self.series_number}.{phase}.tar.lz4"

    def __str__(self) -> str:
        return (
            f"Series({self.accession}.{self.series_number}: {self.series_description})"
        )


@dataclass
class ProcessedSeries:
    """Container for processed series data ready for export"""

    series_info: SeriesInfo
    processing_metadata: Dict[str, Any]
    numpy_volume: Optional[np.ndarray] = None  # For 3D modalities (CT)
    numpy_slices: Optional[List[np.ndarray]] = (
        None  # For 2D modalities (XRay, Mammogram)
    )

    # Optional multi-phase support
    phases: Optional[List["ProcessedSeries"]] = None

    # Backward compatibility
    @property
    def jpeg_slices(self):
        """Backward compatibility for code expecting jpeg_slices"""
        if self.numpy_slices is not None:
            return self.numpy_slices
        # For volumes, return a list of slices
        if self.numpy_volume is not None:
            return [self.numpy_volume[i] for i in range(self.numpy_volume.shape[0])]
        return []

    @property
    def jp2_slices(self):
        """Deprecated: alias to numpy slices for backward compatibility (JPEG2000 removed)."""
        return self.jpeg_slices

    def get_slice_count(self) -> int:
        """Get total number of slices"""
        if self.phases:
            return sum(phase.get_slice_count() for phase in self.phases)
        if self.numpy_volume is not None:
            return self.numpy_volume.shape[0]  # First dimension is slices
        if self.numpy_slices is not None:
            return len(self.numpy_slices)
        return 0

    def get_output_size_mb(self, output_dir: Optional[str] = None) -> float:
        """Get actual output size in MB from existing tarball, or estimate if not available"""
        # If output directory is provided, check for existing tarball
        if output_dir:
            output_path = Path(output_dir) / self.series_info.get_output_filename()
            if output_path.exists():
                try:
                    # Return actual file size
                    return output_path.stat().st_size / (1024 * 1024)
                except (OSError, IOError):
                    return None


@dataclass
class ExportResult:
    """Result of exporting a processed series"""

    series_info: SeriesInfo
    output_path: str
    file_size_mb: float
    slice_count: int
    success: bool
    error_message: Optional[str] = None
    dicom_size_mb: float = 0.0

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.series_info.accession}.{self.series_info.series_number} → {self.output_path} ({self.file_size_mb:.1f}MB, {self.slice_count} slices)"


@dataclass
class PipelineStats:
    """Statistics for pipeline execution"""

    total_series: int
    processed_successfully: int
    failed_series: int
    total_slices: int
    total_output_size_mb: float
    processing_time_seconds: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_series == 0:
            return 0.0
        return (self.processed_successfully / self.total_series) * 100

    def __str__(self) -> str:
        return (
            f"Pipeline Stats:\n"
            f"  Series: {self.processed_successfully}/{self.total_series} successful ({self.success_rate:.1f}%)\n"
            f"  Slices: {self.total_slices:,}\n"
            f"  Output: {self.total_output_size_mb:.1f} MB\n"
            f"  Time: {self.processing_time_seconds:.1f}s"
        )
