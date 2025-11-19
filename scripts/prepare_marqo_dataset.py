"""CLI script to prepare Marqo fashion dataset for indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset
from rich.logging import RichHandler

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


@app.command()
def main(
    output_dir: Annotated[str, typer.Option(help="Output directory for sampled dataset")],
    sample_size: Annotated[int, typer.Option(help="Number of samples to select")] = 3000,
    seed: Annotated[int, typer.Option(help="Random seed for sampling")] = 42,
) -> None:
    """Prepare Marqo fashion dataset: load, rename columns, sample, and save."""
    try:
        logger.info("Loading Marqo/marqo-gs-woman-fashion dataset from HuggingFace...")
        dataset = load_dataset("Marqo/marqo-gs-woman-fashion", split="zero_shot")

        logger.info("Original dataset size: %s", len(dataset))
        logger.info("Original columns: %s", dataset.column_names)

        if "query" not in dataset.column_names:
            msg = "Dataset missing 'query' column"
            raise ValueError(msg)

        if "image" not in dataset.column_names:
            msg = "Dataset missing 'image' column"
            raise ValueError(msg)

        logger.info("Renaming 'query' column to 'text'...")
        dataset = dataset.rename_column("query", "text")

        logger.info("Shuffling dataset with seed=%s...", seed)
        dataset = dataset.shuffle(seed=seed)

        logger.info("Selecting %s samples...", sample_size)
        if len(dataset) < sample_size:
            logger.warning(
                "Dataset has only %s samples, which is less than requested %s. "
                "Using all available samples.",
                len(dataset),
                sample_size,
            )
            sample_size = len(dataset)

        dataset = dataset.select(range(sample_size))

        logger.info("Final dataset size: %s", len(dataset))
        logger.info("Final columns: %s", dataset.column_names)

    except Exception:
        logger.exception("Error preparing Marqo dataset")
        raise typer.Exit(code=1) from None
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = str(output_path)

        logger.info("Saving dataset to %s...", save_path)
        try:
            dataset.save_to_disk(save_path)
        except Exception:
            logger.exception("Failed to save dataset to disk: %s", save_path)
            raise typer.Exit(code=1) from None

        logger.info("✓ Dataset successfully prepared and saved to %s", save_path)
        logger.info("  - Size: %s samples", len(dataset))
        logger.info("  - Columns: %s", dataset.column_names)
        logger.info("  - Seed: %s", seed)


if __name__ == "__main__":
    app()
