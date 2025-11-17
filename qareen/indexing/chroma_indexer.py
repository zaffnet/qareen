from __future__ import annotations

import re

from .base import VectorStoreIndexer
from .exceptions import InvalidCollectionNameError


class ChromaIndexer(VectorStoreIndexer):
    """A vector store indexer for ChromaDB."""

    def index(self, *args: object, **kwargs: object) -> object:
        """Index the dataset."""
        raise NotImplementedError

    def get_collection_name(
        self,
        dataset_name: str,
        environment: str,
        model_id: str,
        alpha: float,
    ) -> str:
        """
        Get the collection name.

        Args:
            dataset_name: The name of the dataset.
            environment: The environment.
            model_id: The ID of the embedding model.
            alpha: The alpha value.

        Returns:
            The collection name.
        """
        name = f"{environment}_{dataset_name}_{model_id}_alpha{alpha:.2f}"
        name_lower = name.lower()
        invalid_chars = list(set(re.findall(r"[^a-z0-9_.-]", name_lower)))
        if invalid_chars:
            raise InvalidCollectionNameError(name, invalid_chars)
        sanitized_name = re.sub(r"[^a-z0-9_.-]", "_", name_lower)
        sanitized_name = re.sub(r"_{2,}", "_", sanitized_name)
        return sanitized_name

    def create_vectorstore(self, *args: object, **kwargs: object) -> object:
        """Create the vector store."""
        raise NotImplementedError

    def get_embeddings(self, *args: object, **kwargs: object) -> object:
        """Get the embeddings."""
        raise NotImplementedError

    def list_available_alphas(self) -> list[float]:
        """List the available alpha values."""
        raise NotImplementedError

    def validate_alpha_available(self, alpha: float) -> bool:
        """Validate that the alpha value is available."""
        raise NotImplementedError
