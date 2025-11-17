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
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Alpha weights (0-1) applied to blend text/image embeddings. "
            "Multiple values produce multiple collections."
        ),
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
    alphas: Sequence[float],
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
        for alpha in alphas:
            collection_name = indexer.get_collection_name(
                dataset_name=loader.get_dataset_name(),
                environment=environment,
                model_id=model_id,
                alpha=alpha,
            )
            print(
                "Preparing to index"
                f" {len(items)} items into collection '{collection_name}'"
                f" with alpha={alpha}"
            )
            # Placeholder for actual indexing logic.
            try:
                indexer.index(items, model_id=model_id, alpha=alpha)
            except NotImplementedError:
                print("Indexing is not implemented yet; this is a structural scaffold.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    models = args.models or settings.default_embedding_models
    alphas = args.alphas or settings.default_alpha_values
    _validate_alpha_values(alphas)
    environment = args.environment or settings.environment
    sample_size = _resolve_sample_size(
        environment=environment,
        cli_sample_size=args.sample_size,
        default_dev_sample_size=settings.dev_sample_size,
    )

    index_dataset(
        dataset_name=args.dataset_name,
        models=models,
        alphas=alphas,
        environment=environment,
        split=args.split,
        sample_size=sample_size,
    )
    return 0


def _validate_alpha_values(alphas: Sequence[float]) -> None:
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha values must be between 0 and 1 inclusive.")


def _resolve_sample_size(
    *,
    environment: str,
    cli_sample_size: int | None,
    default_dev_sample_size: int,
) -> int | None:
    if cli_sample_size is not None:
        return cli_sample_size
    if environment == "dev":
        return default_dev_sample_size
    return None


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
