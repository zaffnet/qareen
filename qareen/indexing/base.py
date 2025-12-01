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
        """Create vector store index.

        Implementations should support a rebuild parameter to control whether
        existing collections are deleted before indexing.

        Args:
            alpha_values: List of alpha values to index
            rebuild: If True, deletes existing collections before indexing
            batch_size: Batch size for processing
            sample_size: Optional sample size override
            environment: Environment (dev/staging/prod)

        Returns:
            Dict mapping alpha values to VectorStore instances
        """
