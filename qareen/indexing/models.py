"""Embedding model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

from langchain_core.embeddings import Embeddings


class EmbeddingModel(ABC):
    """Abstraction over embedding providers (text, image, multimodal)."""

    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        """Load and return the underlying model implementation."""

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> Iterable[list[float]]:
        """Return embeddings for the provided text inputs."""

    @abstractmethod
    def embed_image(self, images: Sequence[Any]) -> Iterable[list[float]]:
        """Return embeddings for the provided image inputs."""

    @abstractmethod
    def embed_multimodal(self, payloads: Sequence[dict[str, Any]]) -> Iterable[list[float]]:
        """Return embeddings for multimodal (text+image) payloads."""

    @abstractmethod
    def as_langchain_embeddings(self) -> Embeddings:
        """Expose the implementation as a LangChain ``Embeddings`` instance."""


__all__ = ["EmbeddingModel"]
