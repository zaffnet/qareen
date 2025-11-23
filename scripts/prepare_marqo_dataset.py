"""CLI script to prepare Marqo fashion dataset for indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset

from qareen.models import Settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    config_file: Annotated[
        Path | None,
        typer.Option(help="Path to configuration file (.env format)"),
    ] = None,
) -> None:
    """Prepare Marqo fashion dataset: load, rename columns, sample, and save.

    Uses configuration from Settings for sample size, seed, and output directory.
    """
    try:
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()

        logger.info("Loading Marqo/marqo-gs-woman-fashion dataset from HuggingFace...")
        dataset = load_dataset("Marqo/marqo-gs-woman-fashion", split="zero_shot")

        logger.info("Original dataset size: %s", len(dataset))
        logger.info("Original columns: %s", dataset.column_names)

        if "query" not in dataset.column_names:
            raise ValueError("Dataset missing 'query' column")
        if "image" not in dataset.column_names:
            raise ValueError("Dataset missing 'image' column")

        logger.info("Renaming 'query' column to 'text'...")
        dataset = dataset.rename_column("query", "text")

        logger.info("Shuffling dataset with seed=%s...", settings.random_seed)
        dataset = dataset.shuffle(seed=settings.random_seed)

        sample_size = settings.dataset_prep_sample_size
        logger.info("Selecting %s samples...", sample_size)

        if len(dataset) < sample_size:
            logger.warning(
                "Dataset has only %s samples (less than requested %s). Using all samples.",
                len(dataset),
                sample_size,
            )
            sample_size = len(dataset)

        dataset = dataset.select(range(sample_size))

        logger.info("Final dataset size: %s", len(dataset))
        logger.info("Final columns: %s", dataset.column_names)

        save_path = str(settings.prepared_dataset_dir)
        logger.info("Saving dataset to %s...", save_path)

        dataset.save_to_disk(save_path)

        logger.info("✅ Dataset successfully prepared and saved")
        logger.info("  - Path: %s", save_path)
        logger.info("  - Size: %s samples", len(dataset))
        logger.info("  - Columns: %s", dataset.column_names)
        logger.info("  - Seed: %s", settings.random_seed)

    except Exception:
        logger.exception("Error preparing Marqo dataset")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
