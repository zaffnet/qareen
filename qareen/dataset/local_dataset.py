"""Local dataset loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from qareen.dataset.base import (
    MISSING_FIELDS_ERROR,
    DatasetLoader,
    extract_dataset_info,
    validate_dataset_schema,
)


class LocalDatasetLoader(DatasetLoader):
    """Loader for datasets saved to disk.

    Loads datasets that were saved using HuggingFace's save_to_disk method.

    Attributes:
        dataset_path: Path to saved dataset directory
        dataset: Loaded dataset instance

    """

    def __init__(self, dataset_path: str | Path) -> None:
        """Initialize local dataset loader.

        Args:
            dataset_path: Path to saved dataset directory

        Raises:
            ValueError: If path does not exist or is not a directory

        """
        if not dataset_path or str(dataset_path).strip() == "":
            msg = "Dataset path is empty"
            raise ValueError(msg)
        path = Path(dataset_path)
        if not path.exists():
            msg = f"Dataset path does not exist: {dataset_path}"
            raise ValueError(msg)
        if not path.is_dir():
            msg = f"Dataset path is not a directory: {dataset_path}"
            raise ValueError(msg)
        self.dataset_path = path
        self._dataset: HFDataset | DatasetDict | None = None

    def load(self) -> HFDataset | DatasetDict:
        """Load dataset from disk.

        Returns:
            Loaded HuggingFace dataset

        """
        if self._dataset is None:
            self._dataset = load_from_disk(str(self.dataset_path))
        return self._dataset

    def validate_schema(self) -> None:
        """Validate dataset has required text and image fields.

        Raises:
            ValueError: If required fields are missing

        """
        dataset = self.load()
        validate_dataset_schema(dataset, MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        """Return dataset identifier.

        Returns:
            Dataset name (directory name)

        """
        return self.dataset_path.name

    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata.

        Returns:
            Dictionary with dataset size, features, and splits

        """
        dataset = self.load()
        return extract_dataset_info(dataset, self.get_dataset_name())
