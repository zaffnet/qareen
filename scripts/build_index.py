from __future__ import annotations

import argparse

from qareen.config.settings import Settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import SigLIPEmbeddingModel


def main():
    settings = Settings()
    parser = argparse.ArgumentParser(description="Build vector store indexes.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="The name of the dataset to index.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=settings.default_embedding_models,
        help="The embedding models to use.",
    )
    parser.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=settings.default_alpha_values,
        help="The alpha values to use for combining embeddings.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=settings.environment,
        help="The environment to build the index in.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Override the dev sample size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="The batch size for indexing.",
    )
    args = parser.parse_args()

    loader = HuggingFaceDatasetLoader(
        args.dataset_name,
        sample_size=args.sample_size,
    )
    loader.load()

    for model_id in args.models:
        model = SigLIPEmbeddingModel(model_id)
        model.load_model()
        for alpha in args.alpha_values:
            indexer = ChromaIndexer()
            collection_name = indexer.get_collection_name(
                dataset_name=args.dataset_name,
                environment=args.environment,
                model_id=model_id,
                alpha=alpha,
            )
            print(f"Indexing collection: {collection_name}")
            # The actual indexing logic will be added here.
            # For now, this just prints the collection name.


if __name__ == "__main__":
    main()
