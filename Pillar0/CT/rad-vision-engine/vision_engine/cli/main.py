"""
Main CLI entry point for vision-engine
"""

import argparse
import sys
import logging
from pathlib import Path

from ..core.config import Config
from ..core.pipeline import VisionPipeline
from ..core.exceptions import VisionEngineError, ConfigurationError


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
        + ([logging.FileHandler(log_file)] if log_file else []),
    )


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Vision-engine: Medical imaging preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CT chest series with default column name
  vision-engine process --config configs/ct_chest.yaml --input-series-csv series_paths.csv --output /output

  # Process with custom path column
  vision-engine process --config configs/ct_chest.yaml --input-series-csv mapping.csv --path-column dest_series_path --output /output

  # Process X-ray series
  vision-engine process --config configs/xray.yaml --input-series-csv xray_series.csv --output /output

  # Process mammography series
  vision-engine process --config configs/mammogram.yaml --input-series-csv mammo_series.csv --output /output
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process DICOM series to JPEG tarballs"
    )

    # Required arguments
    process_parser.add_argument(
        "--config", type=str, required=True, help="YAML configuration file"
    )
    process_parser.add_argument(
        "--input-series-csv",
        type=str,
        required=True,
        help="CSV file with direct paths to series directories",
    )
    process_parser.add_argument(
        "--output", type=str, required=True, help="Output directory for tarballs"
    )
    process_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    process_parser.add_argument(
        "--path-column",
        type=str,
        default="series_path",
        help="Name of the CSV column containing series paths (default: series_path)",
    )

    process_parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop pipeline on first error"
    )
    process_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run (no actual processing)"
    )
    process_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (process limited number of studies)",
    )
    process_parser.add_argument(
        "--debug-limit",
        type=int,
        default=10,
        help="Number of studies to process in debug mode",
    )

    # Global options
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    return parser


def process_command(args):
    """Handle process command"""
    try:
        # Create configuration from YAML with CLI overrides
        cli_args = {
            "input_series_dirs_csv": args.input_series_csv,
            "path_column": args.path_column,
            "workers": args.workers,
            "stop_on_error": args.stop_on_error,
            "dry_run": args.dry_run,
            "debug_mode": args.debug,
            "debug_limit": args.debug_limit,
        }

        config = Config.from_cli_args(cli_args, args.config)

        if args.dry_run:
            print("DRY RUN MODE - No actual processing will occur")
            if config.debug_mode:
                print(f"DEBUG MODE - Limited to {config.debug_max_series} studies")
            print(f"Configuration: {config.to_dict()}")
            return

        if config.debug_mode:
            print(
                f"🐛 DEBUG MODE: Processing limited to {config.debug_max_series} studies"
            )

        # Create and run pipeline
        pipeline = VisionPipeline(config)

        # Run processing
        summary = pipeline.process(args.input_series_csv, args.output)

        # Print summary
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total series: {summary['total_series']}")
        print(f"Processed successfully: {summary['processed_successfully']}")
        print(f"Failed: {summary['failed_series']}")
        print(f"Output directory: {summary['output_directory']}")
        if "mapping_file" in summary and summary["mapping_file"]:
            print(f"Mapping file: {summary['mapping_file']}")

        if summary["failed_details"]:
            print(f"\nFailed series:")
            for series, error in summary["failed_details"][:5]:  # Show first 5 failures
                print(f"  - {series.accession}.{series.series_number}: {error}")
            if len(summary["failed_details"]) > 5:
                print(f"  ... and {len(summary['failed_details']) - 5} more")

        return 0 if summary["failed_series"] == 0 else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Handle commands
    if args.command == "process":
        return process_command(args)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
