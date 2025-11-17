from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader


def main():
    """Download a dataset from HuggingFace and save it to disk."""
    parser = argparse.ArgumentParser(description="Download SQID dataset from HuggingFace.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="zaffnet/SQID-shots-type-1",
        help="The name of the dataset on HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/",
        help="The directory to save the dataset to.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the dataset schema after downloading.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Download only a sample of the dataset.",
    )
    args = parser.parse_args()

    try:
        loader = HuggingFaceDatasetLoader(args.dataset_name, sample_size=args.sample_size)
        dataset = loader.load()
        output_path = args.output_dir / args.dataset_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)

        if args.validate:
            loader.validate_schema()

        print(f"Dataset '{args.dataset_name}' downloaded to '{args.output_dir}'.")
        print("Dataset info:", loader.get_dataset_info())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
