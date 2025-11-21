import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal, cast

from datasets import Dataset
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer, setup_logging
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class SimpleDatasetLoader(DatasetLoader):
    def __init__(self, data):
        self.data = data

    def load(self) -> Any:
        return Dataset.from_dict(self.data)

    def validate_schema(self) -> None:
        pass

    def get_dataset_name(self) -> str:
        return "verify_fusion"

    def get_dataset_info(self) -> dict[str, Any]:
        return {"dataset_name": "verify_fusion", "num_rows": len(self.data["text"])}


def create_image(color):
    return Image.new("RGB", (224, 224), color=color)


def main():
    # Create dummy data
    img_red = create_image("red")
    img_blue = create_image("blue")
    img_green = create_image("green")

    data = {
        "text": ["Red circle", "Blue square", "Green triangle"],
        "image": [img_red, img_blue, img_green],
    }

    loader = SimpleDatasetLoader(data)

    # Use small model
    embedding_model = SIGLIPEmbeddingModel(model_id="google/siglip-base-patch16-224")

    # Settings
    settings = Settings()
    settings.chroma_db_dir = Path("experiments_chroma_db")
    settings.environment = "dev"

    # Clean up
    if os.path.exists(settings.chroma_db_dir):
        shutil.rmtree(settings.chroma_db_dir)

    indexer = ChromaIndexer(loader, embedding_model, settings)

    logger.info("Indexing...")
    # Index with both metrics
    # This will create collections for alpha=0.0, 0.5, 1.0 automatically
    indexer.index(alpha_values=[0.5], rebuild=True, distance_metric="cosine", batch_size=10)
    indexer.index(alpha_values=[0.5], rebuild=True, distance_metric="l2", batch_size=10)

    # Queries
    queries = [
        ("Red", None),  # Text only
        (None, img_blue),  # Image only
    ]

    configs = [
        {"method": "linear", "mode": "similarity", "metric": "cosine", "alpha": 0.5},
        {"method": "rrf", "mode": "similarity", "metric": "cosine", "alpha": 0.5},
        {"method": "linear", "mode": "similarity", "metric": "l2", "alpha": 0.5},
        {"method": "rrf", "mode": "similarity", "metric": "l2", "alpha": 0.5},
        {"method": "linear", "mode": "mmr", "metric": "cosine", "alpha": 0.5},
    ]

    for q_text, q_img in queries:
        logger.info(f"\nQuery: Text='{q_text}', Image={'Present' if q_img else 'None'}")
        for config in configs:
            logger.info(f"  Config: {config}")
            try:
                metric_val = cast(Literal["cosine", "l2"], config["metric"])
                alpha_val = cast(float, config["alpha"])
                method_val = cast(Literal["linear", "rrf"], config["method"])
                mode_val = cast(Literal["similarity", "mmr"], config["mode"])

                vs = indexer.create_vectorstore(
                    "verify_fusion", embedding_model.get_model_id(), alpha_val, "dev", metric_val
                )

                results = indexer.query_multimodal(
                    vectorstore=vs,
                    image=q_img,
                    text=q_text,
                    alpha=alpha_val,
                    k=2,
                    combination_method=method_val,
                    retrieval_mode=mode_val,
                    distance_metric=metric_val,
                )

                for doc, score in results:
                    logger.info(f"    Result: {doc.page_content}, Score: {score:.4f}")
            except Exception as e:
                logger.error(f"    Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
