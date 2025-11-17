"""Script for orchestrating dataset indexing across embedding models."""

from __future__ import annotations

import argparse
from typing import Sequence

from qareen.config import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="HuggingFace dataset identifier or local dataset alias.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to index (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Embedding models to use. Defaults to configuration values when omitted.",
    )
    parser.add_argument(
        "--environment",
        choices=["dev", "staging", "prod"],
        default=None,
        help="Target environment for the collection name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the dev sample size limit.",
    )
    return parser


def index_dataset(
    *,
    dataset_name: str,
    models: Sequence[str],
    environment: str,
    split: str,
    sample_size: int | None,
) -> None:
    loader = HuggingFaceDatasetLoader(dataset_name, split=split)
    items = loader.load()
    if environment == "dev" and sample_size is not None:
        items = items[:sample_size]

    indexer = ChromaIndexer(environment=environment, dev_sample_size=sample_size)
    for model_id in models:
        collection_name = indexer.get_collection_name(
            dataset_name=loader.get_dataset_name(),
            environment=environment,
            model_id=model_id,
        )
        print(f"Preparing to index {len(items)} items into collection '{collection_name}'")
        # Placeholder for actual indexing logic.
        try:
            indexer.index(items, model_id=model_id)
        except NotImplementedError:
            print("Indexing is not implemented yet; this is a structural scaffold.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    models = args.models or settings.default_embedding_models
    environment = args.environment or settings.environment
    sample_size = args.sample_size if args.sample_size is not None else settings.dev_sample_size

    index_dataset(
        dataset_name=args.dataset_name,
        models=models,
        environment=environment,
        split=args.split,
        sample_size=sample_size,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
