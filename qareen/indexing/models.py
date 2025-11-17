"""Embedding model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Provides interface for multimodal embedding generation with text and images.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model with proper caching and device placement."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate L2-normalized text embedding.

        Args:
            text: Input text string

        Returns:
            L2-normalized text embedding vector
        """
        pass

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        """Generate L2-normalized image embedding.

        Args:
            image: PIL Image object or path to image file

        Returns:
            L2-normalized image embedding vector
        """
        pass

    @abstractmethod
    def embed_multimodal(
        self,
        image: Image.Image | str | Path,
        text: str,
        alpha: float,
    ) -> np.ndarray:
        """Generate combined multimodal embedding with alpha weighting.

        Formula: V_combined = Normalize(alpha * V_image + (1 - alpha) * V_text)
        Both V_image and V_text are L2-normalized before combination.

        Args:
            image: PIL Image object or path to image file
            text: Input text string
            alpha: Weight for image embedding (0.0-1.0)

        Returns:
            L2-normalized combined embedding vector
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """Return normalized model identifier.

        Returns:
            Model identifier string
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            Embedding dimension as integer
        """
        pass

    @staticmethod
    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector.

        Args:
            vector: Input vector

        Returns:
            L2-normalized vector

        Raises:
            ValueError: If the input vector has zero norm (zero vector)
        """
        norm: float = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("cannot L2-normalize zero vector")
        return vector / norm
