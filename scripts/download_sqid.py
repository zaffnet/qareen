"""CLI script to download SQID dataset from HuggingFace."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset as HFDataset
from datasets import DatasetDict

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SQID_SUBSET_ERROR = "Expected 'product_image_urls' subset in SQID dataset"


def validate_parquet_file(file_path: Path) -> None:
    """Validate a parquet file by attempting to read it.

    Args:
        file_path: Path to the parquet file to validate

    Raises:
        Exception: If the file cannot be read as a valid parquet file
    """
    try:
        pd.read_parquet(file_path)
    except Exception:
        logger.exception(f"Parquet validation failed for {file_path}")
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted invalid file: {file_path}")
        raise


def download_and_validate_parquet(
    url: str, file_path: Path, timeout: int, max_retries: int
) -> None:
    """Download and validate a parquet file with retry logic.

    Args:
        url: URL to download the parquet file from
        file_path: Path where the file should be saved
        timeout: Timeout in seconds for the download request
        max_retries: Maximum number of retry attempts

    Raises:
        Exception: If download and validation fails after all retries
    """
    for attempt in range(max_retries):
        try:
            if file_path.exists():
                validate_parquet_file(file_path)
                logger.info(f"Successfully validated existing {file_path}")
                break
            else:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                file_path.write_bytes(response.content)
                validate_parquet_file(file_path)
                logger.info(f"Successfully validated {file_path}")
                break
        except (requests.exceptions.RequestException, OSError, ValueError, TypeError):
            if attempt == max_retries - 1:
                logger.exception(
                    f"Failed to download and validate {file_path} after {max_retries} attempts"
                )
                raise
            logger.warning(f"Download/validation failed for {file_path}, retrying...")


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

    parser.add_argument(
        "--combined",
        action="store_true",
        help="Download and combine SQID with ESCI dataset",
    )

    parser.add_argument(
        "--esci-examples-url",
        type=str,
        default="https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        help="URL to ESCI examples parquet file",
    )

    parser.add_argument(
        "--esci-products-url",
        type=str,
        default="https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet",
        help="URL to ESCI products parquet file",
    )

    return parser


def load_and_combine_sqid_esci(
    sqid_dataset_name: str,
    esci_examples_url: str,
    esci_products_url: str,
    cache_dir: Path,
    esci_download_timeout: int | None = None,
) -> HFDataset:
    """Load and combine SQID and ESCI datasets.

    Args:
        sqid_dataset_name: SQID dataset name on HuggingFace
        esci_examples_url: URL to ESCI examples parquet
        esci_products_url: URL to ESCI products parquet
        cache_dir: Cache directory for ESCI files
        esci_download_timeout: Timeout in seconds for downloading ESCI parquet files.
            Defaults to 600, or ESCI_DOWNLOAD_TIMEOUT environment variable if set.

    Returns:
        Combined HuggingFace Dataset
    """
    if esci_download_timeout is None:
        try:
            esci_download_timeout = int(os.getenv("ESCI_DOWNLOAD_TIMEOUT", "600"))
        except ValueError:
            esci_download_timeout = 600
            logger.warning("Invalid ESCI_DOWNLOAD_TIMEOUT environment variable. Using default: 600")

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SQID dataset...")
    sqid_loader = HuggingFaceDatasetLoader(
        dataset_name=sqid_dataset_name, split="train", name="product_image_urls"
    )
    sqid_raw = sqid_loader.load()

    if isinstance(sqid_raw, dict):
        if "product_image_urls" in sqid_raw:
            sqid_dataset = sqid_raw["product_image_urls"]
        else:
            raise ValueError(SQID_SUBSET_ERROR)
    else:
        sqid_dataset = sqid_raw

    logger.info("Loading ESCI dataset...")
    examples_file = cache_dir / "shopping_queries_dataset_examples.parquet"
    products_file = cache_dir / "shopping_queries_dataset_products.parquet"

    max_retries = 3
    download_and_validate_parquet(
        esci_examples_url, examples_file, esci_download_timeout, max_retries
    )
    download_and_validate_parquet(
        esci_products_url, products_file, esci_download_timeout, max_retries
    )

    df_examples = pd.read_parquet(examples_file)
    df_products = pd.read_parquet(products_file)

    logger.info("Filtering ESCI for small_version=1, split='test', product_locale='us'")
    df_examples = df_examples[
        (df_examples["small_version"] == 1)
        & (df_examples["split"] == "test")
        & (df_examples["product_locale"] == "us")
    ]
    df_products = df_products[df_products["product_locale"] == "us"]

    logger.info("Joining ESCI examples with products...")
    df_esci = df_examples.merge(df_products, on=["product_id", "product_locale"], how="left")

    logger.info("Joining with SQID images...")
    df_sqid = sqid_dataset.to_pandas()
    df_combined = df_esci.merge(df_sqid[["product_id", "image_url"]], on="product_id", how="left")

    logger.info("Constructing text fields...")

    def construct_text(row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get("product_title")) and row.get("product_title"):
            parts.append(str(row["product_title"]).strip())
        if pd.notna(row.get("product_description")) and row.get("product_description"):
            parts.append(str(row["product_description"]).strip())
        if pd.notna(row.get("product_bullet_point")) and row.get("product_bullet_point"):
            bp = row["product_bullet_point"]
            if isinstance(bp, str):
                parts.append(bp.strip())
            elif isinstance(bp, list):
                parts.append("\n".join(str(b).strip() for b in bp if b))
        return "\n\n".join(parts) if parts else ""

    df_combined["text"] = df_combined.apply(construct_text, axis=1)

    logger.info("Constructing metadata...")
    metadata_fields = [
        "esci_label",
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "example_id",
        "product_brand",
        "product_color",
    ]
    df_combined["metadata"] = df_combined.apply(
        lambda row: {
            f: row[f] if pd.notna(row.get(f)) else None for f in metadata_fields if f in row
        },
        axis=1,
    )

    df_final = pd.DataFrame(
        {
            "text": df_combined["text"],
            "image": df_combined["image_url"].where(df_combined["image_url"].notna(), None),
            "metadata": df_combined["metadata"],
        }
    )

    both_none = ((df_final["text"] == "") | df_final["text"].isna()) & df_final["image"].isna()
    if both_none.sum() > 0:
        logger.warning(f"Dropping {both_none.sum()} records with both text and image as None")
        df_final = df_final[~both_none]

    logger.info(f"Final combined dataset: {len(df_final)} records")
    return HFDataset.from_pandas(df_final, preserve_index=False)


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
        logger.info(f"Combined mode: {args.combined}")

        if args.combined:
            logger.info("Loading and combining SQID+ESCI datasets...")
            dataset = load_and_combine_sqid_esci(
                sqid_dataset_name=args.dataset_name,
                esci_examples_url=args.esci_examples_url,
                esci_products_url=args.esci_products_url,
                cache_dir=output_dir / "esci_cache",
            )
        else:
            loader: DatasetLoader = HuggingFaceDatasetLoader(
                dataset_name=args.dataset_name,
                split="train",
            )
            dataset = loader.load()

        if args.sample_size:
            logger.info(f"Sampling {args.sample_size} items")
            if isinstance(dataset, DatasetDict):
                sampled_splits = {}
                for split_name, split in dataset.items():
                    sample_count = min(args.sample_size, len(split))
                    sampled_splits[split_name] = split.select(range(sample_count))
                dataset = DatasetDict(sampled_splits)
            else:
                dataset = dataset.select(range(min(args.sample_size, len(dataset))))

        if not args.combined:
            logger.info("Validating schema...")
            loader.validate_schema()

        if isinstance(dataset, DatasetDict):
            total_rows = sum(len(split) for split in dataset.values())
            features = list(next(iter(dataset.values())).features.keys())
        else:
            total_rows = len(dataset)
            features = list(dataset.features.keys())

        logger.info("Dataset info: %d rows, features: %s", total_rows, features)

        if args.combined:
            safe_name = "combined_sqid_esci"
        else:
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.dataset_name)
        save_path = str(output_dir / safe_name)
        dataset.save_to_disk(save_path)
        logger.info(f"Dataset saved to {save_path}")

    except Exception:
        logger.exception("Error downloading dataset")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
