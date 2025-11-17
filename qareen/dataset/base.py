"""Abstract dataset loader contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from .schema import DatasetItem


class DatasetLoader(ABC):
    """Contract for dataset loaders that validate and expose multimodal samples."""

    @abstractmethod
    def load(self) -> Sequence[DatasetItem]:
        """Load the dataset and return validated items."""

    @abstractmethod
    def validate_schema(self, records: Iterable[dict[str, Any]]) -> Sequence[DatasetItem]:
        """Validate raw records against :class:`DatasetSchema`."""

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset identifier used for indexing and provenance."""

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata describing the dataset (e.g., splits, size, description)."""


__all__ = ["DatasetLoader"]
