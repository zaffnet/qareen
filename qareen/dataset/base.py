from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self) -> Any:
        """Load the dataset."""
        raise NotImplementedError

    @abstractmethod
    def validate_schema(self) -> bool:
        """Validate the dataset schema."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Get the dataset name."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Get the dataset info."""
        raise NotImplementedError
