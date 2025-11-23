"""CLI script to visualize comparison of models and alpha values for Marqo dataset."""

from __future__ import annotations

import logging
import os
import random

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from datasets import DatasetDict, load_from_disk
from PIL import Image

from qareen.dataset.local_dataset import LocalDatasetLoader
from qareen.models import Settings
from qareen.retrieving.chroma_retriever import ChromaRetriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to max_length, appending '...' if truncated."""
    return text[:max_length] + "..." if len(text) > max_length else text


def save_query_image(query_image: Image.Image | str | None, output_dir: Path) -> Path | None:
    """Save query image to disk and return the path."""
    if not query_image:
        return None

    query_image_path = output_dir / "query_image.jpg"

    if isinstance(query_image, Image.Image):
        if query_image.mode != "RGB":
            query_image = query_image.convert("RGB")
        query_image.save(query_image_path)
        logger.info("Saved query image to: %s", query_image_path)
        return query_image_path

    return None


def query_all_combinations(
    settings: Settings,
    dataset_name: str,
    query_image: Image.Image | str | None,
    query_text: str,
) -> dict[str, dict[float, list]]:
    """Query all model/alpha combinations and return results."""
    all_results: dict[str, dict[float, list]] = {}

    for model_id in settings.embedding_models:
        logger.info("Processing model: %s", model_id)

        embedding_model = settings.create_embedding_model(model_id)
        retriever = ChromaRetriever(embedding_model, settings)

        all_results[model_id] = {}
        for alpha in settings.alpha_values:
            logger.info("  Querying with alpha=%.3f", alpha)
            try:
                vectorstore = retriever.get_vectorstore(
                    dataset_name=dataset_name,
                    model_id=model_id,
                    alpha=alpha,
                    environment=settings.environment,
                )
                results = retriever.query_multimodal(
                    vectorstore=vectorstore,
                    image=query_image,
                    text=query_text,
                    alpha=alpha,
                    k=settings.k_neighbors,
                )
                all_results[model_id][alpha] = results
            except Exception:
                logger.exception("Failed to query model %s with alpha %.3f", model_id, alpha)
                all_results[model_id][alpha] = []

    return all_results


def generate_markdown(
    output_path: Path,
    settings: Settings,
    sample_idx: int,
    query_text: str,
    query_image_path: Path | None,
    all_results: dict[str, dict[float, list]],
    dataset: Any,
    images_dir: Path,
) -> None:
    """Generate markdown visualization file."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Marqo Fashion Dataset: Model and Alpha Comparison\n\n")

        # Reproducibility information
        f.write("## Reproducibility Information\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Generated | {datetime.now(UTC).isoformat()} |\n")
        f.write(f"| Dataset | `{settings.dataset_path}` |\n")
        f.write(f"| Environment | `{settings.environment}` |\n")
        f.write(f"| K (neighbors) | `{settings.k_neighbors}` |\n")
        f.write(f"| Sample Index | `{sample_idx}` |\n")
        f.write(f"| Random Seed | `{settings.random_seed}` |\n")
        f.write(f"| Models | {len(settings.embedding_models)} |\n")
        f.write(f"| Alpha Values | {len(settings.alpha_values)} |\n\n")
        f.write("---\n\n")

        # Query sample
        f.write("## Query Sample\n\n")
        f.write(f"**Sample Index**: {sample_idx}\n\n")

        if query_image_path and query_image_path.exists():
            f.write(
                f'<div align="center">\n'
                f'<img src="images/{query_image_path.name}" width="400" alt="Query Image">\n'
                f"</div>\n\n"
            )
        else:
            f.write("*No image available*\n\n")

        f.write("**Query Text**:\n\n")
        f.write(f"> {query_text or ''}\n\n")
        f.write("---\n\n")

        # Results for each model/alpha combination
        for model_id in settings.embedding_models:
            f.write(f"## Model: `{model_id}`\n\n")

            for alpha in settings.alpha_values:
                f.write(f"### Alpha = {alpha:.3f}\n\n")

                results = all_results[model_id].get(alpha, [])
                if not results:
                    f.write("*No results available*\n\n")
                    continue

                f.write(
                    "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0;'>\n"
                )

                for i, (doc, score) in enumerate(results, 1):
                    f.write(
                        "<div style='flex: 1; min-width: 180px; max-width: 220px; "
                        "border: 1px solid #ddd; border-radius: 8px; padding: 10px; "
                        "text-align: center; background: #f9f9f9;'>\n"
                    )

                    # Get result image
                    doc_index = doc.metadata.get("index", "unknown")
                    if doc_index != "unknown":
                        try:
                            result_sample = dataset[int(doc_index)]
                            result_image = result_sample.get("image")
                            if result_image and isinstance(result_image, Image.Image):
                                result_img_path = images_dir / f"result_{doc_index}.jpg"
                                if not result_img_path.exists():
                                    if result_image.mode != "RGB":
                                        result_image = result_image.convert("RGB")
                                    result_image.save(result_img_path)
                                f.write(
                                    f'<img src="images/{result_img_path.name}" '
                                    f'width="180" style="border-radius: 4px; margin-bottom: 8px;" '
                                    f'alt="Result {i}"><br>\n'
                                )
                        except (ValueError, IndexError, KeyError):
                            pass

                    f.write(
                        f"<strong>#{i}</strong> "
                        f"<span style='color: #666;'>Score: {score:.3f}</span><br>\n"
                    )
                    preview_text = truncate_text(doc.page_content, max_length=60)
                    f.write(f"<small style='color: #333;'>{preview_text}</small>\n")
                    f.write("</div>\n")

                f.write("</div>\n\n")

            f.write("---\n\n")


@app.command()
def main(
    dataset_path: Annotated[
        str | None,
        typer.Option(help="Dataset path (overrides config)"),
    ] = None,
    sample_index: Annotated[
        int | None,
        typer.Option(help="Specific sample index (overrides random selection)"),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(help="Path to configuration file (.env format)"),
    ] = None,
) -> None:
    """Visualize comparison of models and alpha values for Marqo dataset.

    Uses Settings for all configuration (models, alphas, k, seed, output path).
    """
    try:
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()

        # Override dataset_path if provided
        if dataset_path:
            settings.dataset_path = dataset_path

        if not settings.dataset_path:
            logger.error("dataset_path must be set in config or provided via --dataset-path")
            raise typer.Exit(code=1)

        logger.info("Loading dataset from: %s", settings.dataset_path)
        dataset = load_from_disk(settings.dataset_path)
        dataset_len = len(dataset)
        logger.info("Dataset size: %s", dataset_len)

        if dataset_len == 0:
            logger.error("Dataset is empty. Cannot proceed.")
            raise typer.Exit(code=1)

        # Select sample
        random.seed(settings.random_seed)
        if sample_index is not None:
            if not (0 <= sample_index < dataset_len):
                logger.error("Sample index %s out of bounds [0, %s]", sample_index, dataset_len - 1)
                raise typer.Exit(code=1)
            sample_idx = sample_index
            logger.info("Using specified sample index: %s", sample_idx)
        else:
            sample_idx = random.randint(0, dataset_len - 1)
            logger.info("Randomly selected sample index: %s", sample_idx)

        sample = dataset[sample_idx]
        query_text = sample["text"]
        query_image = sample["image"]

        # Prepare output directories
        images_dir = settings.viz_output_file.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Save query image
        query_image_path = save_query_image(query_image, images_dir)

        # Run queries
        logger.info("Querying all model/alpha combinations...")
        dataset_loader = LocalDatasetLoader(dataset_path=settings.dataset_path)
        dataset_name = dataset_loader.get_dataset_name()

        all_results = query_all_combinations(
            settings=settings,
            dataset_name=dataset_name,
            query_image=query_image,
            query_text=query_text,
        )

        # Generate visualization
        logger.info("Generating markdown visualization at: %s", settings.viz_output_file)
        generate_markdown(
            output_path=settings.viz_output_file,
            settings=settings,
            sample_idx=sample_idx,
            query_text=query_text,
            query_image_path=query_image_path,
            all_results=all_results,
            dataset=dataset,
            images_dir=images_dir,
        )

        logger.info("✅ Visualization saved to: %s", settings.viz_output_file)

    except typer.Exit:
        raise
    except Exception:
        logger.exception("Error generating visualization")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
