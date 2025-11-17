"""Abstract vector store indexing contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qareen.dataset.schema import DatasetItem


class VectorStoreIndexer(ABC):
    """Contract describing how datasets are turned into vector store collections."""

    @abstractmethod
    def index(self, items: Sequence[DatasetItem], *, model_id: str) -> VectorStore:
        """Index the provided dataset items using the supplied embedding model."""

    @abstractmethod
    def get_collection_name(self, dataset_name: str, environment: str, model_id: str) -> str:
        """Return the canonical collection name used by downstream vector stores."""

    @abstractmethod
    def create_vectorstore(self, collection_name: str, *, model_id: str) -> VectorStore:
        """Return the backing vector store instance for the given collection name."""

    @abstractmethod
    def get_embeddings(self, model_id: str) -> Embeddings:
        """Return an embeddings object for the supplied model identifier."""


__all__ = ["VectorStoreIndexer"]
