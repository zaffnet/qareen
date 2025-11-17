"""HuggingFace dataset loader implementation."""

from __future__ import annotations

from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

from qareen.dataset.base import DatasetLoader
from qareen.dataset.schema import DatasetSchema


class HuggingFaceDatasetLoader(DatasetLoader):
    """Loader for HuggingFace datasets.

    Loads datasets from HuggingFace Hub and validates schema compliance.

    Attributes:
        dataset_name: Name/path of the dataset on HuggingFace Hub
        split: Dataset split to load (train/validation/test)
        dataset: Loaded dataset instance
    """

    MISSING_FIELDS_ERROR = (
        "Dataset missing required fields: {missing_fields}. Available fields: {available_fields}"
    )

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        **load_kwargs: Any,
    ) -> None:
        """Initialize HuggingFace dataset loader.

        Args:
            dataset_name: Name/path of dataset on HuggingFace Hub
            split: Dataset split to load
            **load_kwargs: Additional arguments passed to load_dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.load_kwargs = load_kwargs
        self._dataset: HFDataset | DatasetDict | None = None

    def load(self) -> HFDataset | DatasetDict:
        """Load dataset from HuggingFace Hub.

        Returns:
            Loaded HuggingFace dataset
        """
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                **self.load_kwargs,
            )
        return self._dataset

    def validate_schema(self) -> bool:
        """Validate dataset has required text and image fields.

        Returns:
            True if schema is valid

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

        return True

    def get_dataset_name(self) -> str:
        """Return dataset identifier.

        Returns:
            Dataset name
        """
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata.

        Returns:
            Dictionary with dataset size, features, and splits
        """
        dataset = self.load()

        if isinstance(dataset, dict):
            return {
                "dataset_name": self.dataset_name,
                "splits": list(dataset.keys()),
                "num_rows": {k: len(v) for k, v in dataset.items()},
                "features": list(next(iter(dataset.values())).features.keys()),
            }
        else:
            return {
                "dataset_name": self.dataset_name,
                "split": self.split,
                "num_rows": len(dataset),
                "features": list(dataset.features.keys()),
            }
