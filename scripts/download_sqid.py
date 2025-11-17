"""CLI script to download SQID dataset from HuggingFace."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

from qareen.config.settings import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for download script.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Download SQID dataset from HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sqid",
        help="Dataset name/path on HuggingFace Hub",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to config data_dir)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Download only a sample of the dataset",
    )

    return parser


def main() -> int:
    """Main entry point for download script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings = Settings()
        settings.ensure_directories()
        output_dir = args.output_dir or settings.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset: {args.dataset_name}")
        logger.info(f"Output directory: {output_dir}")

        loader = HuggingFaceDatasetLoader(
            dataset_name=args.dataset_name,
            split="train",
        )

        dataset = loader.load()

        if args.sample_size:
            logger.info(f"Sampling {args.sample_size} items")
            dataset = dataset.select(range(min(args.sample_size, len(dataset))))

        logger.info("Validating schema...")
        loader.validate_schema()

        info: dict[str, Any]
        if isinstance(dataset, dict):
            if dataset:
                info = {
                    "dataset_name": loader.get_dataset_name(),
                    "splits": list(dataset.keys()),
                    "num_rows": {k: len(v) for k, v in dataset.items()},
                    "features": list(next(iter(dataset.values())).features.keys()),
                }
            else:
                info = {
                    "dataset_name": loader.get_dataset_name(),
                    "splits": [],
                    "num_rows": {},
                    "features": [],
                }
        else:
            info = {
                "dataset_name": loader.get_dataset_name(),
                "split": loader.split,
                "num_rows": len(dataset),
                "features": list(dataset.features.keys()),
            }
        logger.info(f"Dataset info: {info}")

        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.dataset_name)
        dataset.save_to_disk(output_dir / safe_name)
        logger.info(f"Dataset saved to {output_dir / safe_name}")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}", exc_info=True)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
