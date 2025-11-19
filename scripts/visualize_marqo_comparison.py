"""CLI script to visualize comparison of models and alpha values for Marqo dataset."""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated

import requests
import typer
from datasets import load_from_disk
from PIL import Image
from rich.logging import RichHandler

from qareen.config.settings import Settings
from qareen.dataset.local_dataset import LocalDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
from qareen.indexing.models import EmbeddingModel
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending '...' if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with '...' if needed, or original text
    """
    return text[:max_length] + "..." if len(text) > max_length else text


def download_image(image_url: str, output_path: Path) -> bool:
    """Download image from URL and save to disk.

    Args:
        image_url: URL of the image to download
        output_path: Path where to save the image

    Returns:
        True if successful, False otherwise
    """
    try:
        with requests.get(image_url, timeout=30, stream=True) as response:
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            logger.info(f"Downloaded query image to: {output_path}")
            return True
    except Exception:
        logger.exception(f"Failed to download image from {image_url}")
        return False


@app.command()
def main(
    dataset_path: Annotated[str, typer.Option(help="Path to local dataset directory")],
    models: Annotated[list[str] | None, typer.Option(help="Model IDs to compare")] = None,
    alpha_values: Annotated[
        list[float] | None, typer.Option(help="Alpha values to compare")
    ] = None,
    environment: Annotated[str, typer.Option(help="Environment")] = "dev",
    k: Annotated[int, typer.Option(help="Number of similar items to retrieve")] = 5,
    output: Annotated[
        str, typer.Option(help="Output markdown file path")
    ] = "data/marqo_comparison.md",
    seed: Annotated[int, typer.Option(help="Random seed for query selection")] = 42,
    sample_index: Annotated[
        int | None, typer.Option(help="Specific sample index to use (overrides random selection)")
    ] = None,
) -> None:
    """Visualize comparison of models and alpha values for Marqo dataset.

    Raises:
        typer.Exit: With code 0 for success, 1 for failure
    """
    if environment not in ["dev", "staging", "prod"]:
        logger.error(f"Invalid environment: {environment}. Must be dev, staging, or prod")
        raise typer.Exit(code=1)

    if not isinstance(k, int) or k <= 0:
        logger.error(f"Invalid k value: {k}. Must be an integer > 0")
        raise typer.Exit(code=1)

    if models is not None and not models:
        logger.error("models parameter is an empty list. Must provide at least one model")
        raise typer.Exit(code=1)

    if alpha_values is not None:
        if not alpha_values:
            logger.error(
                "alpha_values parameter is an empty list. Must provide at least one alpha value"
            )
            raise typer.Exit(code=1)
        for alpha in alpha_values:
            if not isinstance(alpha, float) or not (0.0 <= alpha <= 1.0):
                logger.error(f"Invalid alpha value: {alpha}. Must be a float in range [0.0, 1.0]")
                raise typer.Exit(code=1)

    try:
        settings = Settings(environment=environment)

        model_list = models or [
            "openai/clip-vit-large-patch14",
            "Marqo/marqo-fashionSigLIP",
            "google/siglip2-so400m-patch16-512",
            "Marqo/marqo-ecommerce-embeddings-L",
        ]

        alpha_list = alpha_values or [0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.0]

        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        dataset_len = len(dataset)
        logger.info(f"Dataset size: {dataset_len}")

        if dataset_len == 0:
            logger.error("Dataset is empty. Cannot proceed with visualization.")
            raise typer.Exit(code=1)

        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

        if sample_index is not None:
            sample_idx = sample_index
            if not (0 <= sample_idx < dataset_len):
                logger.error(
                    f"Sample index {sample_idx} is out of bounds. "
                    f"Dataset size: {dataset_len}, valid range: [0, {dataset_len - 1}]"
                )
                raise typer.Exit(code=1)
            logger.info(f"Using specified sample index: {sample_idx}")
        else:
            sample_idx = random.randint(0, dataset_len - 1)
            logger.info(f"Randomly selected sample index: {sample_idx}")

        sample = dataset[sample_idx]
        query_text = sample["text"]
        query_image = sample["image"]

        output_path = Path(output)
        images_dir = output_path.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        query_image_path = images_dir / "query_image.jpg"
        if query_image:
            logger.info(f"Saving query image to: {query_image_path}")
            if isinstance(query_image, Image.Image):
                if query_image.mode != "RGB":
                    query_image = query_image.convert("RGB")
                query_image.save(query_image_path)
            elif isinstance(query_image, str):
                download_image(query_image, query_image_path)

        logger.info("Querying all model/alpha combinations...")
        dataset_loader = LocalDatasetLoader(dataset_path=dataset_path)
        dataset_name = dataset_loader.get_dataset_name()

        all_results: dict[str, dict[float, list]] = {}
        for model_id in model_list:
            logger.info(f"Processing model: {model_id}")

            embedding_model: EmbeddingModel
            if model_id.lower().startswith("marqo/"):
                embedding_model = MarqoFashionSigLIPModel(model_id=model_id)
            else:
                embedding_model = SIGLIPEmbeddingModel(model_id=model_id)

            indexer = ChromaIndexer(
                dataset_loader=dataset_loader,
                embedding_model=embedding_model,
                settings=settings,
            )

            all_results[model_id] = {}
            for alpha in alpha_list:
                logger.info(f"  Querying with alpha={alpha:.3f}")
                try:
                    vectorstore = indexer.create_vectorstore(
                        dataset_name=dataset_name,
                        model_id=model_id,
                        alpha=alpha,
                        environment=environment,
                    )
                    results = indexer.query_multimodal(
                        vectorstore=vectorstore,
                        image=query_image,
                        text=query_text,
                        alpha=alpha,
                        k=k,
                    )
                    all_results[model_id][alpha] = results
                except Exception:
                    logger.exception(f"Failed to query model {model_id} with alpha {alpha}")
                    all_results[model_id][alpha] = []

        logger.info(f"Generating markdown visualization at: {output}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Marqo Fashion Dataset: Model and Alpha Comparison\n\n")

            f.write("## Reproducibility Information\n\n")
            f.write(f"**Generated**: {datetime.now(UTC).isoformat()}\n\n")
            f.write(f"**Dataset**: `{dataset_path}`\n")
            f.write(f"**Environment**: `{environment}`\n")
            f.write(f"**K (neighbors)**: `{k}`\n")
            f.write(f"**Sample Index**: `{sample_idx}`\n")
            f.write(f"**Random Seed**: `{seed}`\n")
            f.write(f"**Models**: {len(model_list)}\n")
            f.write(f"**Alpha Values**: {len(alpha_list)}\n\n")

            f.write("---\n\n")

            f.write("## Query Sample\n\n")
            f.write(f"**Sample Index**: {sample_idx}\n\n")

            if query_image_path.exists():
                f.write(
                    f'<img src="images/{query_image_path.name}" width="300" alt="Query Image">\n\n'
                )
            else:
                f.write("*No image available*\n\n")

            f.write("**Query Text**:\n")
            query_text_safe = query_text if query_text is not None else ""
            f.write(f"> {query_text_safe}\n\n")

            f.write("---\n\n")

            for model_id in model_list:
                f.write(f"## Model: `{model_id}`\n\n")

                for alpha in alpha_list:
                    f.write(f"### Alpha = {alpha:.3f}\n\n")

                    results = all_results[model_id].get(alpha, [])

                    if not results:
                        f.write("*No results available*\n\n")
                        continue

                    f.write("<table>\n")
                    f.write("<tr>\n")

                    for i, (doc, score) in enumerate(results, 1):
                        f.write("<td>\n\n")

                        doc_metadata = doc.metadata
                        doc_index = doc_metadata.get("index", "unknown")

                        if doc_index != "unknown":
                            try:
                                result_sample = dataset[int(doc_index)]
                                result_image = result_sample.get("image")
                                if result_image:
                                    result_img_path = images_dir / f"result_{doc_index}.jpg"
                                    if isinstance(result_image, Image.Image):
                                        if not result_img_path.exists():
                                            if result_image.mode != "RGB":
                                                result_image = result_image.convert("RGB")
                                            result_image.save(result_img_path)
                                        f.write(
                                            f'<img src="images/{result_img_path.name}" '
                                            f'width="150" alt="Result {i}"><br>\n'
                                        )
                                    elif isinstance(result_image, str):
                                        f.write(
                                            f'<img src="{result_image}" '
                                            f'width="150" alt="Result {i}"><br>\n'
                                        )
                            except (ValueError, IndexError, KeyError):
                                pass

                        f.write(f"**#{i}** Score: {score:.3f}<br>\n")

                        doc_text = doc.page_content
                        preview_text = truncate_text(doc_text, 80)
                        f.write(f"<small>{preview_text}</small>\n\n")

                        f.write("</td>\n")

                    f.write("</tr>\n")
                    f.write("</table>\n\n")

                f.write("---\n\n")

        logger.info(f"✓ Visualization saved to: {output}")
        logger.info(f"View the file with: open {output}")

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("Error generating visualization")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
