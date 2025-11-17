"""Local dataset loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from qareen.dataset.base import DatasetLoader
from qareen.dataset.schema import DatasetSchema


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

        if isinstance(dataset, dict):
            features = next(iter(dataset.values())).features
        else:
            features = dataset.features

        required_fields = {"text", "image"}
        missing_fields = required_fields - set(features.keys())

        if missing_fields:
            raise ValueError(
                self.MISSING_FIELDS_ERROR.format(
                    missing_fields=missing_fields,
                    available_fields=set(features.keys()),
                )
            )

        if isinstance(dataset, dict):
            non_empty_split = None
            for split in dataset.values():
                if len(split) > 0:
                    non_empty_split = split
                    break
            if non_empty_split is None:
                raise ValueError("Dataset contains no non-empty splits")
            sample = non_empty_split[0]
        else:
            if len(dataset) > 0:
                sample = dataset[0]
            else:
                raise ValueError("Dataset is empty")
        DatasetSchema(**sample)

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
            return {
                "dataset_name": self.get_dataset_name(),
                "splits": list(dataset.keys()),
                "num_rows": {k: len(v) for k, v in dataset.items()},
                "features": list(next(iter(dataset.values())).features.keys()),
            }
        else:
            return {
                "dataset_name": self.get_dataset_name(),
                "num_rows": len(dataset),
                "features": list(dataset.features.keys()),
            }
