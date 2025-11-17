"""Embedding model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Provides interface for multimodal embedding generation with text and images.
    """

    ZERO_VECTOR_ERROR: ClassVar[str] = "cannot L2-normalize zero vector"

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model with proper caching and device placement."""

    @abstractmethod
    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate L2-normalized text embedding.

        Args:
            text: Input text string or None

        Returns:
            L2-normalized text embedding vector or None if text is None
        """

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate L2-normalized image embedding.

        Args:
            image: PIL Image object, path to image file, or None

        Returns:
            L2-normalized image embedding vector or None if image is None
        """

    @abstractmethod
    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
        """Generate combined multimodal embedding with alpha weighting.

        Formula: V_combined = Normalize(alpha * V_image + (1 - alpha) * V_text)
        Both V_image and V_text are L2-normalized before combination.

        Handles missing modalities:
        - If image is None: returns text embedding
        - If text is None: returns image embedding
        - If both are None: raises ValueError

        Args:
            image: PIL Image object, path to image file, or None
            text: Input text string or None
            alpha: Weight for image embedding (0.0-1.0)

        Returns:
            L2-normalized combined embedding vector

        Raises:
            ValueError: If both image and text are None
        """

    @abstractmethod
    def get_model_id(self) -> str:
        """Return normalized model identifier.

        Returns:
            Model identifier string
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            Embedding dimension as integer
        """

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
            raise ValueError(EmbeddingModel.ZERO_VECTOR_ERROR)
        return vector / norm
