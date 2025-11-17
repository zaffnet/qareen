from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStoreIndexer(ABC):
    """Abstract base class for vector store indexers."""

    @abstractmethod
    def index(self, *args: object, **kwargs: object) -> object:
        """Index the dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_collection_name(
        self,
        dataset_name: str,
        environment: str,
        model_id: str,
        alpha: float,
    ) -> str:
        """Get the collection name."""
        raise NotImplementedError

    @abstractmethod
    def create_vectorstore(self, *args: object, **kwargs: object) -> object:
        """Create the vector store."""
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, *args: object, **kwargs: object) -> object:
        """Get the embeddings."""
        raise NotImplementedError

    @abstractmethod
    def list_available_alphas(self) -> list[float]:
        """List the available alpha values."""
        raise NotImplementedError

    @abstractmethod
    def validate_alpha_available(self, alpha: float) -> bool:
        """Validate that the alpha value is available."""
        raise NotImplementedError
