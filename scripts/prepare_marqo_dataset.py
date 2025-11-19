"""CLI script to prepare Marqo fashion dataset for indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def main(
    output_dir: Annotated[
        str, typer.Option(help="Output directory for sampled dataset")
    ] = "data/marqo_fashion_3000",
    sample_size: Annotated[int, typer.Option(help="Number of samples to select")] = 3000,
    seed: Annotated[int, typer.Option(help="Random seed for sampling")] = 42,
) -> int:
    """Prepare Marqo fashion dataset: load, rename columns, sample, and save.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    try:
        logger.info("Loading Marqo/marqo-gs-woman-fashion dataset from HuggingFace...")
        dataset = load_dataset("Marqo/marqo-gs-woman-fashion", split="zero_shot")

        logger.info(f"Original dataset size: {len(dataset)}")
        logger.info(f"Original columns: {dataset.column_names}")

        if "query" not in dataset.column_names:
            logger.error("Dataset missing 'query' column")
            return 1

        if "image" not in dataset.column_names:
            logger.error("Dataset missing 'image' column")
            return 1

        logger.info("Renaming 'query' column to 'text'...")
        dataset = dataset.rename_column("query", "text")

        logger.info(f"Shuffling dataset with seed={seed}...")
        dataset = dataset.shuffle(seed=seed)

        logger.info(f"Selecting {sample_size} samples...")
        if len(dataset) < sample_size:
            logger.warning(
                f"Dataset has only {len(dataset)} samples, which is less than "
                f"requested {sample_size}. Using all available samples."
            )
            sample_size = len(dataset)

        dataset = dataset.select(range(sample_size))

        logger.info(f"Final dataset size: {len(dataset)}")
        logger.info(f"Final columns: {dataset.column_names}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = str(output_path)

        logger.info(f"Saving dataset to {save_path}...")
        dataset.save_to_disk(save_path)

        logger.info(f"✓ Dataset successfully prepared and saved to {save_path}")
        logger.info(f"  - Size: {len(dataset)} samples")
        logger.info(f"  - Columns: {dataset.column_names}")
        logger.info(f"  - Seed: {seed}")

        return 0

    except Exception:
        logger.exception("Error preparing Marqo dataset")
        return 1


if __name__ == "__main__":
    app()
