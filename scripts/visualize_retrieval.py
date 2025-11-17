"""CLI script to visualize similarity search results."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_from_disk

from qareen.config.settings import Settings
from qareen.dataset.local_dataset import LocalDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending '...' if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with '...' if needed, or original text
    """
    return text[:max_length] + "..." if len(text) > max_length else text


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for visualization script.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Visualize similarity search results with images and text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/combined_sqid_esci",
        help="Path to local dataset directory",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Model ID used for indexing",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value used for indexing",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Environment",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar items to retrieve",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/retrieval_visualization.md"),
        help="Output markdown file path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Specific sample index to use (overrides random selection)",
    )

    return parser


def main() -> int:
    """Main entry point for visualization script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings = Settings(environment=args.environment)

        logger.info(f"Loading dataset from: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
        logger.info(f"Dataset size: {len(dataset)}")

        if args.seed is not None:
            random.seed(args.seed)
            logger.info(f"Random seed set to: {args.seed}")

        if args.sample_index is not None:
            sample_idx = args.sample_index
            logger.info(f"Using specified sample index: {sample_idx}")
        else:
            sample_idx = random.randint(0, len(dataset) - 1)
            logger.info(f"Randomly selected sample index: {sample_idx}")

        sample = dataset[sample_idx]
        query_text = sample["text"]
        query_image_url = sample["image"]
        query_metadata = sample.get("metadata", {})

        logger.info("Initializing embedding model and indexer...")
        dataset_loader = LocalDatasetLoader(dataset_path=args.dataset_path)
        embedding_model = SIGLIPEmbeddingModel(model_id=args.model_id)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        logger.info(f"Creating vectorstore for environment={args.environment}, alpha={args.alpha}")
        vectorstore = indexer.create_vectorstore(
            dataset_name=dataset_loader.get_dataset_name(),
            model_id=args.model_id,
            alpha=args.alpha,
            environment=args.environment,
        )

        logger.info(f"Querying for {args.k} similar items...")
        results = vectorstore.similarity_search_with_score(query_text, k=args.k)

        logger.info(f"Generating markdown visualization at: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# Similarity Search Visualization\n\n")

            f.write("## Reproducibility Information\n\n")
            f.write(f"**Generated**: {datetime.now(UTC).isoformat()}\n\n")
            f.write("**Command**:\n```bash\n")
            cmd_parts = [
                "uv run python scripts/visualize_retrieval.py",
                f"--dataset-path {args.dataset_path}",
                f"--model-id {args.model_id}",
                f"--alpha {args.alpha}",
                f"--environment {args.environment}",
                f"--k {args.k}",
                f"--output {args.output}",
            ]
            if args.seed is not None:
                cmd_parts.append(f"--seed {args.seed}")
            if args.sample_index is not None:
                cmd_parts.append(f"--sample-index {args.sample_index}")
            f.write(" \\\n  ".join(cmd_parts))
            f.write("\n```\n\n")

            f.write("**Configuration**:\n")
            f.write(f"- Dataset: `{args.dataset_path}`\n")
            f.write(f"- Model: `{args.model_id}`\n")
            f.write(f"- Alpha: `{args.alpha}`\n")
            f.write(f"- Environment: `{args.environment}`\n")
            f.write(f"- K (neighbors): `{args.k}`\n")
            f.write(f"- Sample Index: `{sample_idx}`\n")
            if args.seed is not None:
                f.write(f"- Random Seed: `{args.seed}`\n")
            f.write("\n---\n\n")

            f.write("## Query Sample\n\n")
            f.write(f"**Sample Index**: {sample_idx}\n\n")

            if query_image_url:
                f.write(f'<img src="{query_image_url}" width="200" alt="Query Image">\n\n')
            else:
                f.write("*No image available*\n\n")

            f.write("**Text**:\n")
            preview_text = truncate_text(query_text, 500)
            f.write(f"```\n{preview_text}\n```\n\n")

            if query_metadata:
                f.write("**Metadata**:\n")
                for key, value in query_metadata.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            f.write("---\n\n")

            f.write(f"## Top {args.k} Similar Results\n\n")

            for i, (doc, score) in enumerate(results, 1):
                f.write(f"### Result #{i} (Score: {score:.4f})\n\n")

                doc_metadata = doc.metadata
                doc_index = doc_metadata.get("index", "unknown")
                has_image = doc_metadata.get("has_image", False)
                has_text = doc_metadata.get("has_text", False)

                f.write(f"**Dataset Index**: {doc_index}\n")
                f.write(f"**Similarity Score**: {score:.4f}\n")
                f.write(f"**Has Image**: {has_image}\n")
                f.write(f"**Has Text**: {has_text}\n\n")

                if has_image and doc_index != "unknown":
                    try:
                        result_sample = dataset[int(doc_index)]
                        result_image_url = result_sample.get("image")
                        result_product_id = result_sample.get("metadata", {}).get(
                            "product_id", "unknown"
                        )
                        if result_image_url:
                            f.write(
                                f'<img src="{result_image_url}" width="200" alt="Result {i}">\n\n'
                            )
                            f.write(f"**Product ID**: `{result_product_id}`\n\n")
                    except (ValueError, IndexError, KeyError):
                        f.write("*Image URL not available*\n\n")
                else:
                    f.write("*No image available*\n\n")

                f.write("**Text**:\n")
                doc_text = doc.page_content
                preview_text = truncate_text(doc_text, 300)
                f.write(f"```\n{preview_text}\n```\n\n")

                if doc_index != "unknown":
                    try:
                        result_sample = dataset[int(doc_index)]
                        result_metadata = result_sample.get("metadata", {})
                        if result_metadata:
                            f.write("**Metadata**:\n")
                            for key, value in result_metadata.items():
                                f.write(f"- **{key}**: {value}\n")
                            f.write("\n")
                    except (ValueError, IndexError, KeyError):
                        pass

                f.write("---\n\n")

        logger.info(f"✓ Visualization saved to: {args.output}")
        logger.info(f"View the file with: open {args.output}")

    except Exception as e:
        logger.exception(f"Error generating visualization: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
