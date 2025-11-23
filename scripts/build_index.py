"""CLI script to build vector store indexes."""

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
        str | None,
        typer.Option(help="Dataset identifier (overrides config)"),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(help="Path to configuration file (.env format)"),
    ] = None,
) -> int:
    """Build vector store indexes for multimodal datasets.

    Reads configuration from Settings (environment variables or config file).
    Only dataset_name is required; all other parameters come from config.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    try:
        # Load settings from config file if provided
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()

        # Override dataset_path if provided via CLI
        if dataset_name:
            settings.dataset_path = dataset_name

        if not settings.dataset_path:
            logger.error("dataset_path must be set in config or provided via --dataset-name")
            return 1

        logger.info("🚀 Building indexes for dataset: %s", settings.dataset_path)
        logger.info("Environment: %s", settings.environment)
        logger.info("Models: %s", settings.embedding_models)
        logger.info("Alpha values: %s", settings.alpha_values)
        logger.info("Batch size: %s", settings.batch_size)
        logger.info("Rebuild: %s", settings.rebuild_collections)

        # Suppress timm deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")

            # Create dataset loader from settings
            dataset_loader = settings.create_dataset_loader()

            logger.info("Loading dataset...")
            dataset_loader.load()
            logger.info("Validating schema...")
            dataset_loader.validate_schema()

            # Determine sample size for dev environment
            sample_size = settings.dev_sample_size if settings.environment == "dev" else None
            if sample_size:
                logger.info("Dev sample size: %s", sample_size)

            # Process each model
            for model_id in settings.embedding_models:
                logger.info("Processing model: %s", model_id)

                embedding_model = settings.create_embedding_model(model_id)

                indexer = ChromaIndexer(
                    dataset_loader=dataset_loader,
                    embedding_model=embedding_model,
                    settings=settings,
                )

                logger.info("Building indexes for alpha values: %s", settings.alpha_values)
                vectorstores = indexer.index(
                    alpha_values=settings.alpha_values,
                    rebuild=settings.rebuild_collections,
                    batch_size=settings.batch_size,
                    sample_size=sample_size,
                    environment=settings.environment,
                )

                # Log completion for each collection
                for alpha in vectorstores:
                    collection_name = get_collection_name(
                        dataset_name=dataset_loader.get_dataset_name(),
                        model_id=embedding_model.get_model_id(),
                        alpha=alpha,
                        environment=settings.environment,
                    )
                    logger.info("✓ Completed: %s (alpha=%.3f)", collection_name, alpha)

    except FileNotFoundError:
        logger.exception("File or directory not found")
        return 1
    except ValueError:
        logger.exception("Invalid configuration or value")
        return 1
    except ValidationError:
        logger.exception("Configuration validation error")
        return 1
    except Exception:
        logger.exception("Unexpected error building index")
        return 1
    else:
        logger.info("✅ All indexes built successfully")
        return 0


if __name__ == "__main__":
    app()
