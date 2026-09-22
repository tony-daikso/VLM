#!/usr/bin/env python
"""Basic usage example for rad-vision-engine (rve)."""

import rve
import numpy as np


def example_load_and_window():
    """Example: Load a sample and apply windowing."""
    print("Example 1: Load and Window\n" + "=" * 40)

    # Load a medical image sample (works with .tar, .tar.gz, .tar.lz4)
    sample = rve.load_sample("path/to/sample.tar.gz")
    print(f"Loaded sample shape: {sample.shape}")

    # Apply a single window
    lung_view = rve.apply_windowing(sample, "lung", "CT")
    print(f"Lung window shape: {lung_view.shape}")

    # Apply multiple windows at once
    multi_window = rve.apply_windowing(sample, ["lung", "bone", "mediastinum"], "CT")
    print(f"Multi-window shape: {multi_window.shape}")

    # Get all available windows for a modality
    available = rve.get_available_windows("CT")
    print(f"Available CT windows: {available}")


def example_process_dicom():
    """Example: Process DICOM files."""
    print("\n\nExample 2: Process DICOM\n" + "=" * 40)

    # Create a processor for CT
    ct_processor = rve.get_processor("CT", config={"target_size": (512, 512)})

    # Create series info
    series_info = rve.SeriesInfo(
        accession="12345",
        series_number="2",
        study_description="CHEST CT",
        series_description="ROUTINE CHEST",
    )

    # Process the series (would need actual DICOM path)
    # processed = ct_processor.process_series(series_info)
    print("CT processor created and ready to process DICOM files")


def example_apply_custom_windows():
    """Example: Apply custom windowing."""
    print("\n\nExample 3: Custom Windowing\n" + "=" * 40)

    # Create synthetic data
    data = np.random.randn(512, 512) * 1000

    # Apply min-max normalization
    normalized = rve.apply_windowing(data, "minmax", "CT")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Apply with custom min/max values
    custom = rve.apply_windowing(data, "minmax", "CT", min_value=-1000, max_value=1000)
    print(
        f"Custom normalized to [-1000, 1000]: [{custom.min():.3f}, {custom.max():.3f}]"
    )

    # Apply all windows for a modality
    all_windows = rve.apply_windowing(data, "all", "CT")
    print(f"All windows shape: {all_windows.shape}")


def example_file_info():
    """Example: Get information about exported files."""
    print("\n\nExample 4: File Information\n" + "=" * 40)

    # Get information about an exported file
    # info = rve.get_export_info("path/to/exported.tar.gz")
    # print(f"Export info: {info}")

    print("Use get_export_info() to retrieve metadata from exported files")


def example_check_exceptions():
    """Example: Handle exceptions."""
    print("\n\nExample 5: Exception Handling\n" + "=" * 40)

    try:
        # Try to load a non-existent file
        rve.load_sample("non_existent_file.tar")
    except rve.VisionEngineError as e:
        print(f"Caught VisionEngineError: {e}")

    try:
        # Try invalid modality
        rve.get_processor("INVALID")
    except ValueError as e:
        print(f"Caught ValueError: {e}")


if __name__ == "__main__":
    print("rad-vision-engine (rve) Basic Usage Examples")
    print("=" * 50)
    print(f"Version: {rve.__version__}")
    print()

    # Run examples that don't require actual data files
    example_apply_custom_windows()
    example_file_info()
    example_check_exceptions()

    print("\n\nFor full examples, provide actual medical image files.")
    print("Install with: pip install -e .")
    print("Then import with: import rve")
