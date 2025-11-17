import argparse
from qareen.config.settings import settings
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel # This will be a mock for now
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Build vector store indexes for a dataset.")
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
        help="A list of model IDs to use for embedding.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=settings.alphas,
        help="A list of alpha values for multimodal embedding.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=settings.environment,
        choices=["dev", "staging", "prod"],
        help="The environment (dev, staging, prod).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.dev_sample_size,
        help="The number of samples to use for the dev environment.",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    loader = HuggingFaceDatasetLoader(
        dataset_name=args.dataset_name,
        sample_size=args.sample_size if args.environment == "dev" else -1,
    )
    dataset = loader.load()

    indexer = ChromaIndexer()

    for model_id in args.models:
        for alpha in args.alphas:
            collection_name = indexer.get_collection_name(
                dataset_name=args.dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=args.environment,
            )
            print(f"Indexing with model '{model_id}' and alpha '{alpha}' into collection '{collection_name}'...")

            # Here we would load the actual model and pass it to the indexer
            # For now, we'll just print a message
            print("Mocking the indexing process.")
            # indexer.index(dataset, model, alpha)


if __name__ == "__main__":
    main()
