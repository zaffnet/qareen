"""Embedding model abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from langchain_core.embeddings import Embeddings
from PIL import Image

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Provides interface for multimodal embedding generation with text and images.
    """

    ZERO_VECTOR_ERROR: ClassVar[str] = "cannot L2-normalize zero vector"
    ZERO_NORM_TOLERANCE: ClassVar[float] = 1e-8

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model with proper caching and device placement."""

    @abstractmethod
    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate L2-normalized text embedding."""

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate L2-normalized image embedding."""

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
        """Return normalized model identifier."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""

    @staticmethod
    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector.

        Raises:
            ValueError: If the input vector is empty or has zero norm (zero vector)
        """
        if vector.size == 0:
            raise ValueError("cannot L2-normalize empty vector")
        norm: float = float(np.linalg.norm(vector))
        if norm <= EmbeddingModel.ZERO_NORM_TOLERANCE:
            raise ValueError(EmbeddingModel.ZERO_VECTOR_ERROR)
        return vector / norm


class EmbeddingModelWrapper(Embeddings):
    """LangChain Embeddings wrapper for EmbeddingModel.

    Wraps the embedding model for use with LangChain's Chroma integration.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """Initialize wrapper.

        Args:
            embedding_model: Embedding model instance for query embedding
        """
        self.embedding_model = embedding_model
        self._embedding_dim: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using the embedding model.

        Args:
            texts: Input texts to embed

        Returns:
            List of embedding vectors as lists of floats

        Raises:
            RuntimeError: If embedding dimension cannot be determined or embedding fails
            ValueError: If embedding model returns None for any text
        """
        if not texts:
            return []
        if self._embedding_dim is None:
            try:
                self._embedding_dim = self.embedding_model.embedding_dim
            except (AttributeError, RuntimeError, TypeError) as e:
                logger.exception("Failed to get embedding dimension")
                raise RuntimeError("Cannot determine embedding dimension") from e

        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_model.embed_text(text)
                if embedding is None:
                    model_id = self.embedding_model.get_model_id()
                    raise ValueError(
                        f"Embedding returned None for text. "
                        f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
                    )
                embedding_list = cast("list[float]", embedding.tolist())
                if len(embedding_list) != self._embedding_dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                        f"got {len(embedding_list)}",
                    )
                embeddings.append(embedding_list)
            except ValueError:
                raise
            except Exception as e:
                logger.exception(f"Failed to embed text: {text[:100] if text else 'None'}...")
                raise RuntimeError("Embedding failed for text") from e

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using the embedding model.

        Args:
            text: Query text to embed

        Returns:
            Text embedding vector as list of floats

        Raises:
            ValueError: If embedding is None or cannot be converted to list
        """
        embedding = self.embedding_model.embed_text(text)
        model_id = self.embedding_model.get_model_id()
        if embedding is None:
            raise ValueError(
                f"Embedding returned None for provided text. "
                f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
            )
        if hasattr(embedding, "tolist"):
            return cast("list[float]", embedding.tolist())
        if hasattr(embedding, "__iter__") and not isinstance(embedding, (str, bytes)):
            return cast("list[float]", list(embedding))
        raise ValueError(
            f"Unsupported embedding format for model {model_id}. "
            f"Expected array-like object with tolist() method or iterable, "
            f"got {type(embedding).__name__}. Text snippet: {text[:50] if text else 'None'}...",
        )
