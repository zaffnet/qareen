"""HuggingFace dataset loader implementation."""

from __future__ import annotations

from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

from qareen.dataset.base import (
    MISSING_FIELDS_ERROR,
    DatasetLoader,
    extract_dataset_info,
    validate_dataset_schema,
)


class HuggingFaceDatasetLoader(DatasetLoader):
    """Loader for HuggingFace datasets.

    Loads datasets from HuggingFace Hub and validates schema compliance.

    Attributes:
        dataset_name: Name/path of the dataset on HuggingFace Hub
        split: Dataset split to load (train/validation/test)
        dataset: Loaded dataset instance
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        **load_kwargs: object,
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
        """Load dataset from HuggingFace Hub."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                **self.load_kwargs,
            )
        return self._dataset

    def validate_schema(self) -> None:
        """Validate dataset has required text and image fields."""
        dataset = self.load()
        validate_dataset_schema(dataset, MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        """Return dataset identifier."""
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        dataset = self.load()
        return extract_dataset_info(dataset, self.dataset_name, self.split)
