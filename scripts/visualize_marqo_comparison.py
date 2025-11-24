from __future__ import annotations

import logging
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from datasets import load_from_disk
from PIL import Image

from qareen.dataset.local_dataset import LocalDatasetLoader
from qareen.models import Settings
from qareen.retrieving.chroma_retriever import ChromaRetriever

os.environ["ANONYMIZED_TELEMETRY"] = "False"


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def truncate_text(text: str, max_length: int = 80) -> str:
    """
    Truncates text to at most max_length characters and appends "..." when truncation occurs.
    
    Parameters:
        text (str): The string to truncate.
        max_length (int): Maximum allowed length of the returned string; must be greater than or equal to 0.
    
    Returns:
        str: The original text if its length is less than or equal to max_length, otherwise a truncated string ending with "...".
    """
    return text[:max_length] + "..." if len(text) > max_length else text


def save_query_image(query_image: Image.Image | str | None, output_dir: Path) -> Path | None:
    """
    Save an RGB query image to disk as "query_image.jpg" and return its path.
    
    If the provided image is not in RGB mode it will be converted before saving. If `query_image` is None or not a PIL Image, nothing is saved and None is returned.
    
    Parameters:
        query_image (PIL.Image.Image | str | None): The query image to save; must be a PIL Image instance to be saved.
        output_dir (pathlib.Path): Directory where the image file `query_image.jpg` will be written.
    
    Returns:
        pathlib.Path | None: Path to the saved `query_image.jpg` when saved, or `None` if no valid image was provided.
    """
    if not query_image or not isinstance(query_image, Image.Image):
        return None
    query_image_path = output_dir / "query_image.jpg"
    if query_image.mode != "RGB":
        query_image = query_image.convert("RGB")
    query_image.save(query_image_path)
    return query_image_path


def query_all_combinations(
    settings: Settings,
    dataset_name: str,
    query_image: Image.Image | str | None,
    query_text: str,
) -> dict[str, dict[float, list]]:
    """
    Collects multimodal retrieval results across all configured embedding models and alpha values.
    
    Queries each embedding model in settings for the given dataset and query (image and text) for every alpha in settings.alpha_values and aggregates the results into a nested mapping keyed first by model_id then by alpha.
    
    Parameters:
        settings (Settings): Configuration object containing embedding_models, alpha_values, k_neighbors, environment, and helper factories.
        dataset_name (str): Name of the dataset to load vectorstores for.
        query_image (PIL.Image.Image | str | None): Query image (PIL Image or image path) or None when no image query is used.
        query_text (str): Query text.
    
    Returns:
        dict[str, dict[float, list]]: A mapping from model_id to a mapping of alpha to the list of retrieval results for that model/alpha. Each list contains the raw results returned by the retriever (typically tuples of document and score). If a query for a particular model/alpha fails, that alpha maps to an empty list.
    """
    all_results: dict[str, dict[float, list]] = {}
    for model_id in settings.embedding_models:
        logger.info("Processing model: %s", model_id)
        embedding_model = settings.create_embedding_model(model_id)
        retriever = ChromaRetriever(embedding_model, settings)
        all_results[model_id] = {}
        for alpha in settings.alpha_values:
            try:
                vectorstore = retriever.get_vectorstore(
                    dataset_name=dataset_name,
                    model_id=model_id,
                    alpha=alpha,
                    environment=settings.environment,
                )
                all_results[model_id][alpha] = retriever.query_multimodal(
                    vectorstore=vectorstore,
                    image=query_image,
                    text=query_text,
                    alpha=alpha,
                    k=settings.k_neighbors,
                )
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
    """
    Write a Markdown/HTML report comparing embedding models and alpha values for a single query sample and save it to `output_path`.
    
    Parameters:
        output_path (Path): File path to write the generated Markdown/HTML report.
        settings (Settings): Configuration containing dataset_path, environment, k_neighbors, random_seed, embedding_models, and alpha_values used to annotate the report.
        sample_idx (int): Index of the query sample within the dataset.
        query_text (str): Text query for the sample (rendered in the report).
        query_image_path (Path | None): Path to the saved query image to embed in the report, or `None` if no query image is available.
        all_results (dict[str, dict[float, list]]): Nested mapping of results organized as { model_id: { alpha: [(doc, score), ...] } } where `doc` objects expose `metadata` and `page_content`.
        dataset (Any): Sequence-like dataset where items can be indexed by integer and may contain an "image" (PIL Image) for result rendering.
        images_dir (Path): Directory where result images will be saved and referenced from the report.
    
    Side effects:
        - Writes the report to `output_path`.
        - May save result images into `images_dir` for embedding in the report.
    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Marqo Fashion Dataset: Model and Alpha Comparison\n\n")
        f.write("## Reproducibility Information\n\n")
        f.write("| Parameter | Value |\n|-----------|-------|\n")
        f.write(f"| Generated | {datetime.now(UTC).isoformat()} |\n")
        f.write(f"| Dataset | `{settings.dataset_path}` |\n")
        f.write(f"| Environment | `{settings.environment}` |\n")
        f.write(f"| K (neighbors) | `{settings.k_neighbors}` |\n")
        f.write(f"| Sample Index | `{sample_idx}` |\n")
        f.write(f"| Random Seed | `{settings.random_seed}` |\n")
        f.write(f"| Models | {len(settings.embedding_models)} |\n")
        f.write(f"| Alpha Values | {len(settings.alpha_values)} |\n\n")
        f.write("---\n\n## Query Sample\n\n")
        f.write(f"**Sample Index**: {sample_idx}\n\n")
        if query_image_path and query_image_path.exists():
            img_tag = f'<img src="images/{query_image_path.name}" width="400" alt="Query Image">'
            f.write(f'<div align="center">\n{img_tag}\n</div>\n\n')
        else:
            f.write("*No image available*\n\n")
        f.write(f"**Query Text**:\n\n> {query_text or ''}\n\n---\n\n")

        for model_id in settings.embedding_models:
            f.write(f"## Model: `{model_id}`\n\n")
            for alpha in settings.alpha_values:
                f.write(f"### Alpha = {alpha:.3f}\n\n")
                results = all_results[model_id].get(alpha, [])
                if not results:
                    f.write("*No results available*\n\n")
                    continue
                f.write(
                    "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0;'>\n",
                )
                for i, (doc, score) in enumerate(results, 1):
                    div_style = (
                        "<div style='flex: 1; min-width: 180px; max-width: 220px; "
                        "border: 1px solid #ddd; border-radius: 8px; padding: 10px; "
                        "text-align: center; background: #f9f9f9;'>\n"
                    )
                    f.write(div_style)
                    doc_index = doc.metadata.get("index", "unknown")
                    if doc_index != "unknown":
                        try:
                            result_image = dataset[int(doc_index)].get("image")
                            if result_image and isinstance(result_image, Image.Image):
                                result_img_path = images_dir / f"result_{doc_index}.jpg"
                                if not result_img_path.exists():
                                    if result_image.mode != "RGB":
                                        result_image = result_image.convert("RGB")
                                    result_image.save(result_img_path)
                                img_tag = (
                                    f'<img src="images/{result_img_path.name}" width="180" '
                                    f'style="border-radius: 4px; margin-bottom: 8px;" '
                                    f'alt="Result {i}"><br>\n'
                                )
                                f.write(img_tag)
                        except (ValueError, IndexError, KeyError):
                            # Skip if image path is invalid or not found
                            pass
                    score_text = (
                        f"<strong>#{i}</strong> "
                        f"<span style='color: #666;'>Score: {score:.3f}</span><br>\n"
                    )
                    f.write(score_text)
                    preview = truncate_text(doc.page_content, max_length=60)
                    f.write(f"<small style='color: #333;'>{preview}</small>\n</div>\n")
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
    """
    Generate a markdown visualization comparing retrieval results across configured embedding models and alpha values.
    
    Loads settings (optionally from a .env file), loads the dataset, selects a sample (by index or randomly), queries all model/alpha combinations for that sample, and writes a Markdown/HTML report and associated images to the configured visualization output path. Exits with a non-zero code via typer.Exit on invalid configuration, missing/empty dataset, or other fatal errors.
    
    Parameters:
        dataset_path (str | None): Optional path to the dataset that overrides the value in the configuration.
        sample_index (int | None): Optional zero-based index of the sample to visualize; if omitted, a random sample is chosen using the configured random seed.
        config_file (Path | None): Optional path to a .env configuration file used to construct Settings.
    """
    try:
        settings = Settings(_env_file=str(config_file)) if config_file else Settings()
        if dataset_path:
            settings.dataset_path = dataset_path
        if not settings.dataset_path:
            logger.error("dataset_path must be set in config or provided via --dataset-path")
            raise typer.Exit(code=1)

        logger.info("Loading dataset from: %s", settings.dataset_path)
        dataset = load_from_disk(settings.dataset_path)
        dataset_len = len(dataset)
        if dataset_len == 0:
            logger.error("Dataset is empty. Cannot proceed.")
            raise typer.Exit(code=1)

        random.seed(settings.random_seed)
        if sample_index is not None:
            if not (0 <= sample_index < dataset_len):
                logger.error("Sample index %s out of bounds [0, %s]", sample_index, dataset_len - 1)
                raise typer.Exit(code=1)
            sample_idx = sample_index
        else:
            sample_idx = random.randint(0, dataset_len - 1)

        sample = dataset[sample_idx]
        images_dir = settings.viz_output_file.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Querying all model/alpha combinations...")
        dataset_loader = LocalDatasetLoader(dataset_path=settings.dataset_path)
        all_results = query_all_combinations(
            settings=settings,
            dataset_name=dataset_loader.get_dataset_name(),
            query_image=sample["image"],
            query_text=sample["text"],
        )

        logger.info("Generating markdown visualization at: %s", settings.viz_output_file)
        generate_markdown(
            output_path=settings.viz_output_file,
            settings=settings,
            sample_idx=sample_idx,
            query_text=sample["text"],
            query_image_path=save_query_image(sample["image"], images_dir),
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