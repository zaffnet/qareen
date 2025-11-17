from __future__ import annotations

import argparse
from pathlib import Path

from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader


def main():
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

    loader = HuggingFaceDatasetLoader(args.dataset_name, sample_size=args.sample_size)
    dataset = loader.load()
    dataset.save_to_disk(args.output_dir / args.dataset_name)

    if args.validate:
        loader.validate_schema()

    print(f"Dataset '{args.dataset_name}' downloaded to '{args.output_dir}'.")
    print("Dataset info:", loader.get_dataset_info())


if __name__ == "__main__":
    main()
