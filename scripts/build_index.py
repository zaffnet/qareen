"""CLI script to build vector store indexes."""

from __future__ import annotations

import argparse
import logging
import sys

from qareen.config.settings import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for build_index script.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Build vector store indexes for multimodal datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset identifier",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model IDs (default: from config)",
    )

    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values for embedding combination (default: from config)",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Environment (dev/staging/prod)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override dev sample size",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )

    return parser


def main() -> int:
    """Main entry point for build_index script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings = Settings(environment=args.environment.lower())
        settings.ensure_directories()

        models = args.models or settings.embedding_models
        models = list(dict.fromkeys(models))

        alpha_values = args.alpha_values or settings.alpha_values
        alpha_values = sorted(list(set(alpha_values)))

        for alpha in alpha_values:
            if not (0.0 <= alpha <= 1.0):
                logger.error(f"Alpha value {alpha} must be in range [0.0, 1.0]")
                return 1

        logger.info(f"Building indexes for dataset: {args.dataset_name}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Models: {models}")
        logger.info(f"Alpha values: {alpha_values}")
        logger.info(f"Batch size: {args.batch_size}")

        sample_size = None
        if settings.environment == "dev":
            sample_size = args.sample_size or settings.dev_sample_size
            logger.info(f"Dev sample size: {sample_size}")

        dataset_loader = HuggingFaceDatasetLoader(
            dataset_name=args.dataset_name,
            split="train",
        )

        logger.info("Loading dataset...")
        dataset_loader.load()
        logger.info("Validating schema...")
        dataset_loader.validate_schema()

        for model_id in models:
            logger.info(f"Processing model: {model_id}")

            embedding_model = SIGLIPEmbeddingModel(model_id=model_id)

            indexer = ChromaIndexer(
                dataset_loader=dataset_loader,
                embedding_model=embedding_model,
                settings=settings,
            )

            logger.info(f"Building indexes for alpha values: {alpha_values}")
            vectorstores = indexer.index(
                alpha_values=alpha_values,
                batch_size=args.batch_size,
                sample_size=sample_size,
            )

            for alpha, _vectorstore in vectorstores.items():
                collection_name = indexer.get_collection_name(
                    dataset_name=args.dataset_name,
                    model_id=embedding_model.get_model_id(),
                    alpha=alpha,
                    environment=settings.environment,
                )
                logger.info(f"✓ Completed collection: {collection_name} (alpha={alpha:.2f})")

    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        return 1
    else:
        logger.info("✓ All indexes built successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
