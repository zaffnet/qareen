from qareen.dataset import DatasetItem, DatasetLoader, HuggingFaceDatasetLoader, LocalDatasetLoader
from qareen.indexing import ChromaIndexer, EmbeddingModel, SIGLIPEmbeddingModel, VectorStoreIndexer
from qareen.models import Settings

__version__ = "0.1.0"

__all__ = [
    "ChromaIndexer",
    "DatasetItem",
    "DatasetLoader",
    "EmbeddingModel",
    "HuggingFaceDatasetLoader",
    "LocalDatasetLoader",
    "SIGLIPEmbeddingModel",
    "Settings",
    "VectorStoreIndexer",
    "check_gpu_available",
]


def check_gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
