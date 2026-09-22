"""
rad-vision-engine (rve): A high-performance medical image processing engine.

This module provides a clean API for medical image processing tasks including:
- Loading medical images from various formats (DICOM, tarballs)
- Applying windowing transformations
- Multi-modality support (CT, X-ray, Mammography)
"""

# Core data loading functionality
from vision_engine.utils.data_loader import (
    load_vision_sample as load_sample,
    get_export_info,
    load_nifti,
)

# Windowing functionality
from vision_engine.utils.windowing_utils import (
    apply_windowing,
    apply_multiple_windows,
    get_available_windows,
    apply_anatomical_window,
    batch_apply_windowing_vectorized,
    ANATOMICAL_WINDOWS,
    PERCENTILE_WINDOWS,
)

# Core data structures
from vision_engine.core.data_structures import (
    SeriesInfo,
    ProcessedSeries,
    ExportResult,
)

# Configuration
from vision_engine.core.config import Config

# Exceptions
from vision_engine.core.exceptions import (
    VisionEngineError,
    ConfigurationError,
    ProcessingError,
    ExportError,
)

# Version
__version__ = "1.0.0"


# Convenient access to modality-specific processors
def get_processor(modality: str, config=None):
    """Get a processor instance for the specified modality.

    Args:
        modality: One of 'CT', 'XR', 'MG'
        config: Optional configuration object or dictionary.
               Can be a Config instance or a simple dict/object with 'processing' attribute

    Returns:
        Processor instance
    """
    from vision_engine.processing.ct_processor import CTProcessor
    from vision_engine.processing.xray_processor import XRayProcessor
    from vision_engine.processing.mammogram_processor import MammogramProcessor
    from vision_engine.processing.mr_processor import MRProcessor

    processors = {
        "CT": CTProcessor,
        "XR": XRayProcessor,
        "MG": MammogramProcessor,
        "MR": MRProcessor,
    }

    if modality not in processors:
        raise ValueError(
            f"Unknown modality: {modality}. Must be one of {list(processors.keys())}"
        )

    # Create a minimal config-like object if none provided
    if config is None:
        # Create a simple namespace object that processors expect
        class SimpleConfig:
            def __init__(self):
                self.processing = {"target_size": [512, 512]}
                # Default anatomy based on modality
                default_anatomy = {"CT": "chest", "XR": "chest", "MG": "breast"}
                self.anatomy = default_anatomy.get(modality, "unknown")

        config = SimpleConfig()

    return processors[modality](config)


# Convenient access to exporters
def get_exporter(format: str, config: dict):
    """Get an exporter instance for the specified format.

    Args:
        format: One of 'lz4', 'video', 'torch', 'hevc_image'
        config: Configuration dictionary

    Returns:
        Exporter instance
    """
    from vision_engine.lz4_exporter import LZ4Exporter
    from vision_engine.video_exporter import VideoExporter
    from vision_engine.torch_exporter import TorchExporter
    from vision_engine.hevc_image_exporter import HEVCImageExporter

    exporters = {
        "lz4": LZ4Exporter,
        "video": VideoExporter,
        "torch": TorchExporter,
        "hevc_image": HEVCImageExporter,
    }

    if format not in exporters:
        raise ValueError(
            f"Unknown format: {format}. Must be one of {list(exporters.keys())}"
        )

    return exporters[format](config)


# Main exports
__all__ = [
    # Data loading
    "load_sample",
    "get_export_info",
    "load_nifti",
    # Windowing
    "apply_windowing",
    "apply_multiple_windows",
    "get_available_windows",
    "apply_anatomical_window",
    "ANATOMICAL_WINDOWS",
    "PERCENTILE_WINDOWS",
    # Data structures
    "SeriesInfo",
    "ProcessedSeries",
    "ExportResult",
    # Configuration
    "Config",
    # Exceptions
    "VisionEngineError",
    "ConfigurationError",
    "ProcessingError",
    "ExportError",
    # Helper functions
    "get_processor",
    "get_exporter",
]
