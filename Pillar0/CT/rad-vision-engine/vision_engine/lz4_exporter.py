"""
LZ4 exporter that creates compressed tarballs from processed series.
Optimized for fast decompression in deep learning training pipelines.
"""

import os
import io
import tarfile
import logging
from typing import List, Dict, Any
from pathlib import Path
import lz4.frame
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np

from .core.data_structures import ProcessedSeries, ExportResult, SeriesInfo
from .core.exceptions import ExportError
from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)


class LZ4Exporter(BaseExporter):
    """Export processed series as LZ4 compressed tarballs with raw numpy arrays"""

    def __init__(self, config):
        super().__init__(config)

        # Get LZ4-specific configuration with safe defaults
        lz4_config = self.exporter_config.get("lz4", {})
        self.compression_level = int(lz4_config.get("compression_level", 1))
        # Force internal compression mode to 'lz4' for correct file naming and metadata
        self.compression = "lz4"

    def _get_output_filename(self, series_info: SeriesInfo) -> str:
        """Always use .tar.lz4 for LZ4 exporter regardless of global config."""
        base_name = f"{series_info.accession}.{series_info.series_number}.0"
        return f"{base_name}.tar.lz4"

    def export_tarballs(
        self, processed_series_list: List[ProcessedSeries], output_dir: str
    ) -> Dict[str, Any]:
        """Export multiple processed series as tarballs"""
        logger.info(
            f"Exporting {len(processed_series_list)} series to {output_dir} with LZ4 compression"
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export in parallel
        start_time = time.time()
        export_results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_series = {
                executor.submit(self._export_single_series, series, output_dir): series
                for series in processed_series_list
            }

            for future in as_completed(future_to_series):
                try:
                    result = future.result()
                    export_results.append(result)
                    logger.info(str(result))
                except Exception as e:
                    series = future_to_series[future]
                    error_result = ExportResult(
                        series_info=series.series_info,
                        output_path="",
                        file_size_mb=0,
                        slice_count=0,
                        success=False,
                        error_message=str(e),
                        dicom_size_mb=0.0,
                    )
                    export_results.append(error_result)
                    logger.error(f"Export failed for {series.series_info}: {e}")

        # Calculate summary statistics
        successful_exports = [r for r in export_results if r.success]
        total_size_mb = sum(r.file_size_mb for r in successful_exports)
        total_slices = sum(r.slice_count for r in successful_exports)
        processing_time = time.time() - start_time

        # Calculate average compression ratio
        total_dicom_size_mb = sum(
            getattr(r, "dicom_size_mb", 0) for r in successful_exports
        )
        if total_dicom_size_mb > 0:
            compression_ratio = total_size_mb / total_dicom_size_mb
            compression_percent = (1 - compression_ratio) * 100
            logger.info(
                f"DICOM->LZ4 compression: {total_dicom_size_mb:.1f}MB → {total_size_mb:.1f}MB "
                f"(ratio: {compression_ratio:.3f}, {compression_percent:.1f}% reduction)"
            )
        else:
            logger.warning(
                "Could not calculate compression ratio (DICOM sizes unavailable)"
            )

        summary = {
            "total_series": len(processed_series_list),
            "successful_exports": len(successful_exports),
            "failed_exports": len(export_results) - len(successful_exports),
            "total_output_size_mb": total_size_mb,
            "total_slices": total_slices,
            "processing_time_seconds": processing_time,
            "export_results": export_results,
        }

        logger.info(
            f"Export completed: {len(successful_exports)}/{len(processed_series_list)} successful"
        )
        logger.info(f"Total output: {total_size_mb:.1f} MB, {total_slices:,} slices")

        return summary

    def export(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        """Export a single processed series (for pipeline compatibility)"""
        return self._export_single_series(processed_series, output_dir)

    def _export_single_series(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        """Export a single processed series as tarball"""
        series_info = processed_series.series_info

        # Calculate original DICOM size
        dicom_size_mb = self._calculate_dicom_size_mb(series_info)

        # Generate output filename
        output_path = output_dir / self._get_output_filename(series_info)

        # Skip if already exists
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            return ExportResult(
                series_info=series_info,
                output_path=str(output_path),
                file_size_mb=file_size_mb,
                slice_count=len(processed_series.jp2_slices),
                success=True,
                error_message="Skipped (already exists)",
                dicom_size_mb=dicom_size_mb,
            )

        try:
            # Create tarball in memory
            tar_buffer = io.BytesIO()

            # Create uncompressed tar first
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                # Check if we have a volume (3D) or slices (2D)
                if processed_series.numpy_volume is not None:
                    # Store entire volume as single numpy array
                    npy_buffer = io.BytesIO()
                    np.save(npy_buffer, processed_series.numpy_volume)
                    npy_data = npy_buffer.getvalue()

                    # Add to tarball
                    tarinfo = tarfile.TarInfo(name="volume.npy")
                    tarinfo.size = len(npy_data)
                    tar.addfile(tarinfo, io.BytesIO(npy_data))

                elif processed_series.numpy_slices is not None:
                    # Store individual slices (for 2D modalities like XRay, Mammogram)
                    for i, slice_array in enumerate(processed_series.numpy_slices):
                        npy_buffer = io.BytesIO()
                        np.save(npy_buffer, slice_array)
                        npy_data = npy_buffer.getvalue()

                        # Add to tarball
                        tarinfo = tarfile.TarInfo(name=f"{i:04d}.npy")
                        tarinfo.size = len(npy_data)
                        tar.addfile(tarinfo, io.BytesIO(npy_data))
                else:
                    raise ValueError(
                        "ProcessedSeries must have either numpy_volume or numpy_slices"
                    )

                # Add metadata file
                metadata_json = self._create_metadata_json(processed_series)
                metadata_bytes = metadata_json.encode("utf-8")

                tarinfo = tarfile.TarInfo(name="metadata.json")
                tarinfo.size = len(metadata_bytes)
                tar.addfile(tarinfo, io.BytesIO(metadata_bytes))

            # Compress with LZ4
            tar_data = tar_buffer.getvalue()
            compressed_data = lz4.frame.compress(
                tar_data,
                compression_level=self.compression_level,
                block_size=lz4.frame.BLOCKSIZE_MAX4MB,  # Larger blocks for better compression
                block_linked=True,  # Better compression ratio
                content_checksum=True,  # Data integrity
                return_bytearray=False,
            )

            # Write compressed data to disk
            with open(output_path, "wb") as f:
                f.write(compressed_data)

            # Calculate file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            return ExportResult(
                series_info=series_info,
                output_path=str(output_path),
                file_size_mb=file_size_mb,
                slice_count=len(processed_series.jp2_slices),
                success=True,
                dicom_size_mb=dicom_size_mb,
            )

        except Exception as e:
            raise ExportError(f"Failed to export series {series_info}: {e}")

    def _create_metadata_json(self, processed_series: ProcessedSeries) -> str:
        """Create metadata JSON for the tarball"""
        import json
        import sys
        from datetime import datetime

        # Get the original DICOM directory path
        dicom_source_path = processed_series.series_info.get_dicom_path()

        # Capture command line arguments used to run vision-engine
        command_line_args = " ".join(sys.argv)

        # Get current timestamp
        timestamp = datetime.now().isoformat()

        metadata = {
            "series_info": {
                "accession": processed_series.series_info.accession,
                "series_uid": processed_series.series_info.series_uid,
                "series_number": processed_series.series_info.series_number,
                "series_description": processed_series.series_info.series_description,
                "modality": processed_series.series_info.modality,
                "anatomy": processed_series.series_info.anatomy,
                "slice_thickness": processed_series.series_info.slice_thickness,
                "acquisition_time": processed_series.series_info.acquisition_time,
            },
            "source_data": {
                "dicom_directory": dicom_source_path,
                "original_slice_count": processed_series.series_info.slice_count,
            },
            "processing_metadata": processed_series.processing_metadata,
            "export_settings": {
                "format": "numpy",  # Raw numpy arrays instead of JPEG2000
                "compression": "lz4",
                "lz4_compression_level": self.compression_level,
                "bit_depth": 16,
                "exported_slice_count": len(processed_series.jp2_slices),
            },
            "generation_info": {
                "timestamp": timestamp,
                "command_line": command_line_args,
                "configuration": self._get_relevant_config(),
            },
        }

        return json.dumps(metadata, indent=2)

    def _get_relevant_config(self) -> Dict[str, Any]:
        """Extract relevant configuration settings for metadata"""
        return {
            "modality": getattr(self.config, "modality", None),
            "anatomy": getattr(self.config, "anatomy", None),
            "processing": self.config.processing,
            "export": {},  # Removed legacy field
            "debug_mode": getattr(self.config, "debug_mode", False),
            "debug_limit": getattr(self.config, "debug_limit", None),
            "dry_run": getattr(self.config, "dry_run", False),
        }
