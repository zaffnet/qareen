"""CLI script to build vector store indexes."""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError
from rich.logging import RichHandler

from qareen.config.settings import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.dataset.local_dataset import LocalDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

if TYPE_CHECKING:
    from qareen.dataset.base import DatasetLoader
    from qareen.indexing.models import EmbeddingModel


class Environment(str, Enum):
    """Environment options for indexing."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

app = typer.Typer()


def create_embedding_model(model_id: str) -> EmbeddingModel:
    """Create embedding model from model ID.

    Args:
        model_id: Model identifier
            (e.g., "Marqo/marqo-fashionSigLIP", "google/siglip...")

    Returns:
        Instantiated embedding model

    """
    with warnings.catch_warnings():
        timm_msg = ".*Importing from timm.models.layers is deprecated.*"
        warnings.filterwarnings("ignore", message=timm_msg)
        if model_id.lower().startswith("marqo/"):
            return MarqoFashionSigLIPModel(model_id=model_id)
        return SIGLIPEmbeddingModel(model_id=model_id)


@app.command()
def main(
    dataset_name: Annotated[str, typer.Option(help="Dataset identifier")],
    models: Annotated[
        list[str] | None, typer.Option(help="Model IDs (default: from config)")
    ] = None,
    alpha_values: Annotated[
        list[float] | None,
        typer.Option(help="Alpha values for embedding combination (default: from config)"),
    ] = None,
    environment: Annotated[
        Environment, typer.Option(help="Environment (dev/staging/prod)")
    ] = Environment.DEV,
    sample_size: Annotated[int | None, typer.Option(help="Override dev sample size")] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 3000,
    rebuild: Annotated[
        bool, typer.Option(help="Delete existing collections before indexing")
    ] = False,
) -> int:
    """Build vector store indexes for multimodal datasets.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        settings = Settings(environment=environment.value)
        settings.ensure_directories()

        model_list = models or settings.embedding_models
        model_list = list(set(model_list))

        alpha_list = alpha_values or settings.alpha_values
        alpha_list = sorted(set(alpha_list))

        for alpha in alpha_list:
            if not (0.0 <= alpha <= 1.0):
                logger.error("Alpha value %s must be in range [0.0, 1.0]", alpha)
                return 1

        logger.info("Building indexes for dataset: %s", dataset_name)
        logger.info("Environment: %s", environment.value)
        logger.info("Models: %s", model_list)
        logger.info("Alpha values: %s", alpha_list)
        logger.info("Batch size: %s", batch_size)

        dev_sample_size = None
        if settings.environment == "dev":
            dev_sample_size = sample_size or settings.dev_sample_size
            logger.info("Dev sample size: %s", dev_sample_size)

        dataset_loader: DatasetLoader
        dataset_path = Path(dataset_name)
        if dataset_path.exists():
            if not dataset_path.is_dir():
                raise ValueError(
                    f"Dataset path exists but is not a directory: {dataset_name}. "
                    "Please provide a directory path or remove the file to use HuggingFace Hub."
                )
            logger.info("Loading dataset from local path: %s", dataset_name)
            dataset_loader = LocalDatasetLoader(dataset_path=dataset_name)
        else:
            logger.info("Loading dataset from HuggingFace Hub: %s", dataset_name)
            dataset_loader = HuggingFaceDatasetLoader(
                dataset_name=dataset_name,
                split="train",
            )

        logger.info("Loading dataset...")
        dataset_loader.load()
        logger.info("Validating schema...")
        dataset_loader.validate_schema()

        for model_id in model_list:
            logger.info("Processing model: %s", model_id)

            embedding_model = create_embedding_model(model_id)

            indexer = ChromaIndexer(
                dataset_loader=dataset_loader,
                embedding_model=embedding_model,
                settings=settings,
            )

            logger.info("Building indexes for alpha values: %s", alpha_list)
            vectorstores = indexer.index(
                alpha_values=alpha_list,
                rebuild=rebuild,
                batch_size=batch_size,
                sample_size=dev_sample_size,
            )

            for alpha, _vectorstore in vectorstores.items():
                dataset_name_for_collection = dataset_loader.get_dataset_name()
                collection_name = indexer.get_collection_name(
                    dataset_name=dataset_name_for_collection,
                    model_id=embedding_model.get_model_id(),
                    alpha=alpha,
                    environment=settings.environment,
                )
                logger.info("✓ Completed collection: %s (alpha=%.3f)", collection_name, alpha)

    except FileNotFoundError:
        logger.exception("File or directory not found")
        return 1
    except ValueError:
        logger.exception("Invalid value or configuration")
        return 1
    except ValidationError:
        logger.exception("Validation error")
        return 1
    except Exception:
        logger.exception("Unexpected error building index")
        return 1
    else:
        logger.info("✓ All indexes built successfully")
        return 0


if __name__ == "__main__":
    app()
