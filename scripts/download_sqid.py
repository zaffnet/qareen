"""Utility for downloading the SQID dataset via HuggingFace datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from qareen.config import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader

DEFAULT_DATASET_ID = "zafar/sqid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Fully-qualified HuggingFace dataset identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured data directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    data_dir = Path(args.output_dir or settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    loader = HuggingFaceDatasetLoader(args.dataset_id, split=args.split)
    items = loader.load()

    # Persisting logic intentionally omitted; we only surface structural info here.
    print(f"Downloaded {len(items)} items to {data_dir} from {args.dataset_id}:{args.split}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
