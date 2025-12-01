from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from datasets import Dataset, load_dataset

from qareen.models import Settings

REQUIRED_DATASET_COLUMNS = {"query", "image"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


class MissingColumnsError(ValueError):
    """Exception raised when required dataset columns are missing."""

    def __init__(self, required: set[str], missing: set[str]) -> None:
        """Initialize exception with required and missing columns.

        Args:
            required: Set of required column names.
            missing: Set of missing column names.

        """
        required_sorted = sorted(required)
        missing_sorted = sorted(missing)
        message = (
            f"Dataset missing columns: {', '.join(missing_sorted)} "
            f"(required: {', '.join(required_sorted)})"
        )
        super().__init__(message)
        self.required = required
        self.missing = missing


def validate_dataset_columns(dataset: Dataset, required_columns: set[str]) -> None:
    """Validate that the dataset contains the required columns.

    Args:
        dataset: The dataset to check.
        required_columns: A set of column names that must be present.

    Raises:
        MissingColumnsError: If any required columns are missing.

    """
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise MissingColumnsError(required_columns, missing)


@app.command()
def main(
    config_file: Annotated[
        Path | None,
        typer.Option(help="Path to configuration file (.env format)"),
    ] = None,
) -> None:
    try:
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()

        logger.info("Loading Marqo/marqo-gs-woman-fashion dataset from HuggingFace...")
        dataset = load_dataset("Marqo/marqo-gs-woman-fashion", split="zero_shot")

        validate_dataset_columns(dataset, REQUIRED_DATASET_COLUMNS)

        dataset = dataset.rename_column("query", "text").shuffle(seed=settings.random_seed)

        sample_size = min(settings.dataset_prep_sample_size, len(dataset))
        if len(dataset) < settings.dataset_prep_sample_size:
            logger.warning(
                "Dataset has only %s samples (less than requested %s). Using all samples.",
                len(dataset),
                settings.dataset_prep_sample_size,
            )

        dataset = dataset.select(range(sample_size))
        save_path = str(settings.prepared_dataset_dir)
        dataset.save_to_disk(save_path)

        logger.info("✅ Dataset successfully prepared and saved")
        logger.info(
            "  - Path: %s | Size: %s samples | Seed: %s",
            save_path,
            len(dataset),
            settings.random_seed,
        )

    except Exception:
        logger.exception("Error preparing Marqo dataset")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
