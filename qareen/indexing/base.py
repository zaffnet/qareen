"""Abstract base class for vector store indexers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qareen.indexing.exceptions import (
    AlphaNotAvailableError,
    CollectionNameTooLongError,
)


class VectorStoreIndexer(ABC):
    """Abstract base class for vector store indexing.

    Provides common functionality for collection naming and validation.
    Implementations handle specific vector store backends.
    """

    @abstractmethod
    def index(self, *args: Any, **kwargs: Any) -> VectorStore | dict[float, VectorStore]:
        """Create vector store index.

        Returns:
            LangChain VectorStore instance or dict of VectorStore instances
        """
        pass

    @abstractmethod
    def create_vectorstore(self, *args: Any, **kwargs: Any) -> VectorStore:
        """Create LangChain VectorStore instance.

        Returns:
            VectorStore instance for the collection
        """
        pass

    @abstractmethod
    def get_embeddings(self, *args: Any, **kwargs: Any) -> Embeddings:
        """Return LangChain Embeddings instance.

        Returns:
            Embeddings instance for the model
        """
        pass

    def get_collection_name(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float | None = None,
        environment: str = "dev",
    ) -> str:
        """Generate sanitized collection name.

        Format: {environment}_{dataset_name}_{model_id}_alpha{alpha_value}

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value (optional, formatted to 2 decimals if provided)
            environment: Environment (dev/staging/prod)

        Returns:
            Sanitized collection name

        Raises:
            CollectionNameTooLongError: If name exceeds 63 characters
        """
        parts = [
            environment.lower(),
            dataset_name,
            model_id,
        ]

        if alpha is not None:
            parts.append(f"alpha{alpha:.2f}")

        name = "_".join(parts)

        name = name.lower()
        name = re.sub(r"[^a-z0-9_\-]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        if len(name) > 63:
            raise CollectionNameTooLongError(
                collection_name=name,
                max_length=63,
            )

        return name

    def list_available_alphas(
        self,
        dataset_name: str,
        model_id: str,
        environment: str = "dev",
    ) -> list[float]:
        """List available alpha values for a dataset/model combination.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            environment: Environment (dev/staging/prod)

        Returns:
            Sorted list of available alpha values
        """
        return []

    def validate_alpha_available(
        self,
        alpha: float,
        dataset_name: str,
        model_id: str,
        environment: str = "dev",
    ) -> None:
        """Validate alpha value is available (has been indexed).

        Args:
            alpha: Alpha value to check
            dataset_name: Dataset identifier
            model_id: Model identifier
            environment: Environment (dev/staging/prod)

        Raises:
            AlphaNotAvailableError: If alpha has not been indexed
        """
        available = self.list_available_alphas(dataset_name, model_id, environment)

        normalized_alpha = round(alpha, 2)
        normalized_available = [round(a, 2) for a in available]

        if normalized_alpha not in normalized_available:
            raise AlphaNotAvailableError(
                alpha=alpha,
                available_alphas=available,
                model_id=model_id,
                dataset_name=dataset_name,
                environment=environment,
            )
