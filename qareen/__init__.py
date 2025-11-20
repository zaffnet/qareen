"""qareen: A tool for analyzing and optimizing multimodal few-shot example selection."""

import os
import warnings

from qareen.config import Settings
from qareen.dataset import (
    DatasetItem,
    DatasetLoader,
    DatasetSchema,
    HuggingFaceDatasetLoader,
    LocalDatasetLoader,
)
from qareen.indexing import (
    AlphaMismatchError,
    AlphaNotAvailableError,
    ChromaIndexer,
    CollectionNameTooLongError,
    EmbeddingModel,
    InvalidAlphaError,
    InvalidCollectionNameError,
    SIGLIPEmbeddingModel,
    VectorStoreIndexer,
)

__version__ = "0.1.0"

__all__ = [
    "AlphaMismatchError",
    "AlphaNotAvailableError",
    "ChromaIndexer",
    "CollectionNameTooLongError",
    "DatasetItem",
    "DatasetLoader",
    "DatasetSchema",
    "EmbeddingModel",
    "HuggingFaceDatasetLoader",
    "InvalidAlphaError",
    "InvalidCollectionNameError",
    "LocalDatasetLoader",
    "SIGLIPEmbeddingModel",
    "Settings",
    "VectorStoreIndexer",
    "check_gpu_available",
]


def check_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available for PyTorch operations.

    This function performs a lazy check, importing torch only when called.
    It respects the QAREEN_SUPPRESS_GPU_WARNING environment variable to suppress
    warnings when CUDA is not available.

    Returns:
        bool: True if CUDA is available, False otherwise.

    Example:
        >>> if check_gpu_available():
        ...     device = "cuda"
        ... else:
        ...     device = "cpu"
    """
    try:
        import torch

        is_available = torch.cuda.is_available()

        if not is_available:
            suppress_warning = os.getenv("QAREEN_SUPPRESS_GPU_WARNING", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if not suppress_warning:
                warnings.warn(
                    "CUDA is not available. For GPU support, please install a CUDA-enabled "
                    "PyTorch build from https://pytorch.org/get-started/locally/ before "
                    "installing qareen. The package will work with CPU-only PyTorch, but "
                    "GPU acceleration will not be available.",
                    UserWarning,
                    stacklevel=2,
                )

        return is_available
    except ImportError:
        # torch not installed, which is fine - it may be absent due to
        # environment or optional dependency
        return False
