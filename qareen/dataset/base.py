"""Abstract base class for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetLoader(ABC):
    """Abstract base class for loading datasets in standardized format.

    Implementations must provide methods to load, validate, and get metadata
    about datasets.
    """

    @abstractmethod
    def load(self) -> Any:
        """Load dataset and return in standardized format.

        Returns:
            Dataset in standardized format (implementation-specific)
        """
        pass

    @abstractmethod
    def validate_schema(self) -> None:
        """Validate dataset has required fields (text, image).

        Raises:
            ValueError: If required fields are missing
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return dataset identifier.

        Returns:
            Dataset name/identifier
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata.

        Returns:
            Dictionary containing dataset metadata (size, splits, etc.)
        """
        pass
