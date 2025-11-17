"""CLI script to download all images and create a gallery visualization."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from datasets import load_from_disk
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _positive_float(value: str) -> float:
    """Parse and validate a positive float value.

    Args:
        value: String value to parse

    Returns:
        Parsed float value

    Raises:
        argparse.ArgumentTypeError: If value is not a positive float
    """
    try:
        fvalue = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float") from err
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be > 0, got {fvalue}")
    return fvalue


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for image gallery script.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Download all images and create gallery visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/combined_sqid_esci",
        help="Path to local dataset directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory to save downloaded images",
    )

    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("data/image_gallery.md"),
        help="Output markdown file path",
    )

    parser.add_argument(
        "--images-per-row",
        type=int,
        default=10,
        help="Number of images per row in gallery",
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=100,
        help="Width of each image in pixels",
    )

    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Request timeout in seconds (allows float values, must be > 0). "
        "Increase for slow connections.",
    )

    return parser


def main() -> int:
    """Main entry point for image gallery script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        logger.info(f"Loading dataset from: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
        logger.info(f"Dataset size: {len(dataset)}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Images will be saved to: {args.output_dir}")

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)

        logger.info("Downloading images...")
        downloaded_images = []
        failed_downloads = 0

        with requests.Session() as session:
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            for idx in tqdm(range(len(dataset)), desc="Downloading images"):
                sample = dataset[idx]
                image_url = sample.get("image")
                product_id = sample.get("metadata", {}).get("product_id", f"unknown_{idx}")

                if not image_url:
                    logger.debug(f"No image URL for index {idx}, product_id {product_id}")
                    failed_downloads += 1
                    continue

                image_filename = f"{product_id}.jpg"
                image_path = args.output_dir / image_filename

                if image_path.exists():
                    downloaded_images.append((idx, image_url, product_id, image_filename))
                    continue

                try:
                    response = session.get(image_url, timeout=args.timeout)
                    response.raise_for_status()
                    image_path.write_bytes(response.content)
                    downloaded_images.append((idx, image_url, product_id, image_filename))
                except requests.RequestException as e:
                    logger.debug(f"Failed to download image for {product_id}: {e}")
                    failed_downloads += 1
                except OSError as e:
                    logger.debug(f"Failed to write image file for {product_id}: {e}")
                    failed_downloads += 1

        logger.info(f"Downloaded {len(downloaded_images)} images, {failed_downloads} failed")

        logger.info(f"Generating gallery markdown at: {args.output_markdown}")

        markdown_parent = args.output_markdown.parent.resolve()
        images_dir = args.output_dir.resolve()
        try:
            image_rel_path = Path(os.path.relpath(images_dir, markdown_parent))
        except ValueError:
            image_rel_path = images_dir

        with open(args.output_markdown, "w") as f:
            f.write("# Image Gallery\n\n")
            f.write(f"**Total Images**: {len(downloaded_images)}\n")
            f.write(f"**Images per row**: {args.images_per_row}\n")
            f.write(f"**Image width**: {args.image_width}px\n\n")
            f.write("---\n\n")

            for i in range(0, len(downloaded_images), args.images_per_row):
                row_images = downloaded_images[i : i + args.images_per_row]

                for _idx, _image_url, product_id, image_filename in row_images:
                    local_image_path = image_rel_path / image_filename
                    f.write(
                        f'<img src="{local_image_path}" width="{args.image_width}" '
                        f'alt="{product_id}" style="margin: 5px;"> '
                    )
                f.write("\n\n")

                for _idx, _image_url, product_id, _image_filename in row_images:
                    f.write(f"<sub>{product_id}</sub> ")
                f.write("\n\n")

                f.write("---\n\n")

        logger.info(f"✓ Gallery saved to: {args.output_markdown}")
        logger.info(f"✓ Images saved to: {args.output_dir}")
        logger.info(f"View the gallery with: open {args.output_markdown}")

    except (OSError, requests.RequestException) as e:
        logger.exception(f"Error generating image gallery: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error generating image gallery: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
