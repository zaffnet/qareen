"""Local dataset loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from qareen.dataset.base import DatasetLoader, validate_dataset_schema


class LocalDatasetLoader(DatasetLoader):
    """Loader for datasets saved to disk.

    Loads datasets that were saved using HuggingFace's save_to_disk method.

    Attributes:
        dataset_path: Path to saved dataset directory
        dataset: Loaded dataset instance
    """

    MISSING_FIELDS_ERROR = (
        "Dataset missing required fields: {missing_fields}. Available fields: {available_fields}"
    )

    def __init__(self, dataset_path: str | Path) -> None:
        """Initialize local dataset loader.

        Args:
            dataset_path: Path to saved dataset directory
        """
        self.dataset_path = Path(dataset_path)
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
        validate_dataset_schema(dataset, self.MISSING_FIELDS_ERROR)

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

        if isinstance(dataset, dict):
            if not dataset:
                features: list[str] = []
            else:
                features = list(next(iter(dataset.values())).features.keys())
            return {
                "dataset_name": self.get_dataset_name(),
                "splits": list(dataset.keys()),
                "num_rows": {k: len(v) for k, v in dataset.items()},
                "features": features,
            }
        else:
            return {
                "dataset_name": self.get_dataset_name(),
                "num_rows": len(dataset),
                "features": list(dataset.features.keys()),
            }
