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

    @abstractmethod
    def create_vectorstore(self, *args: Any, **kwargs: Any) -> VectorStore:
        """Create LangChain VectorStore instance.

        Returns:
            VectorStore instance for the collection
        """

    @abstractmethod
    def get_embeddings(self, *args: Any, **kwargs: Any) -> Embeddings:
        """Return LangChain Embeddings instance.

        Returns:
            Embeddings instance for the model
        """

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
            ValueError: If dataset_name, model_id, or environment validation fails
            CollectionNameTooLongError: If name exceeds 63 characters
        """
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise ValueError("dataset_name must be a non-empty string")  # noqa: TRY003

        model_id = model_id.strip()
        if not model_id:
            raise ValueError("model_id must be a non-empty string")  # noqa: TRY003

        environment = environment.strip()
        env = environment.lower()
        if env not in ("dev", "staging", "prod"):
            raise ValueError(  # noqa: TRY003
                f"environment must be one of 'dev', 'staging', or 'prod', got '{environment}'"
            )

        sanitized_parts = []
        for part in [env, dataset_name, model_id]:
            sanitized = part.lower()
            sanitized = re.sub(r"[^a-z0-9_]+", "_", sanitized)
            sanitized = re.sub(r"_+", "_", sanitized)
            sanitized = sanitized.strip("_")
            sanitized_parts.append(sanitized)

        base_name = "_".join(sanitized_parts)

        if alpha is not None:
            alpha_suffix = f"alpha{alpha:.2f}".lower()
            name = f"{base_name}_{alpha_suffix}"
        else:
            name = base_name

        if len(name) > 63:
            raise CollectionNameTooLongError(
                collection_name=name,
                max_length=63,
            )

        return name

    @abstractmethod
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
