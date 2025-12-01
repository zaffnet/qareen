from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.models import Settings
from qareen.utils.naming import get_collection_name

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    dataset_name: Annotated[
        str | None, typer.Option(help="Dataset identifier (overrides config)")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option(help="Path to configuration file (.env format)")
    ] = None,
) -> int:
    try:
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()
        if dataset_name:
            settings.dataset_path = dataset_name
        if not settings.dataset_path:
            logger.error("dataset_path must be set in config or provided via --dataset-name")
            return 1

        logger.info("🚀 Building indexes for dataset: %s", settings.dataset_path)
        logger.info(
            "Environment: %s | Models: %s | Alpha values: %s | Batch size: %s | Rebuild: %s",
            settings.environment,
            settings.embedding_models,
            settings.alpha_values,
            settings.batch_size,
            settings.rebuild_collections,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
            dataset_loader = settings.create_dataset_loader()
            dataset_loader.load()
            dataset_loader.validate_schema()

            sample_size = settings.dev_sample_size if settings.environment == "dev" else None
            if sample_size:
                logger.info("Dev sample size: %s", sample_size)

            for model_id in settings.embedding_models:
                logger.info("Processing model: %s", model_id)
                embedding_model = settings.create_embedding_model(model_id)
                indexer = ChromaIndexer(
                    dataset_loader=dataset_loader,
                    embedding_model=embedding_model,
                    settings=settings,
                )

                vectorstores = indexer.index(
                    alpha_values=settings.alpha_values,
                    rebuild=settings.rebuild_collections,
                    batch_size=settings.batch_size,
                    sample_size=sample_size,
                    environment=settings.environment,
                )

                for alpha in vectorstores:
                    name = get_collection_name(
                        dataset_loader.get_dataset_name(),
                        embedding_model.get_model_id(),
                        alpha,
                        settings.environment,
                    )
                    logger.info("✓ Completed: %s (alpha=%.3f)", name, alpha)

        logger.info("✅ All indexes built successfully")
        return 0
    except (FileNotFoundError, ValueError, ValidationError, Exception) as e:
        logger.exception("Error: %s", type(e).__name__)
        return 1


if __name__ == "__main__":
    app()
