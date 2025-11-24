"""Abstract base class for vector store indexers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreIndexer(ABC):
    """Abstract base class for vector store indexing.

    Provides common functionality for collection naming and validation.
    Implementations handle specific vector store backends.
    """

    @abstractmethod
    def index(
        self,
        alpha_values: list[float],
        *,
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
        environment: str = "dev",
    ) -> dict[float, Any]:
        """
        Index embeddings for the given alpha values into backend vector stores.
        
        Parameters:
            alpha_values (list[float]): Alpha values to index.
            rebuild (bool): If True, delete existing collections for each alpha before indexing.
            batch_size (int): Number of items to process per batch.
            sample_size (int | None): Optional limit on the number of items to index; if None, index all available.
            environment (str): Target environment identifier (e.g., "dev", "staging", "prod").
        
        Returns:
            dict[float, Any]: Mapping from each alpha value to the corresponding backend store instance.
        """