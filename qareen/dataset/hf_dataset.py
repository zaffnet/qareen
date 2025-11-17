from __future__ import annotations

from typing import Any

from datasets import load_dataset

from .base import DatasetLoader
from .schema import DatasetItem


class HuggingFaceDatasetLoader(DatasetLoader):
    """A dataset loader for HuggingFace datasets."""

    def __init__(self, dataset_name: str, sample_size: int | None = None):
        """
        Initialize the HuggingFaceDatasetLoader.

        Args:
            dataset_name: The name of the dataset on HuggingFace.
            sample_size: The number of samples to load.
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.dataset = None
        self._schema_validated = False

    def load(self) -> Any:
        """Load the dataset from HuggingFace."""
        self.dataset = load_dataset(self.dataset_name, split="train")
        assert self.dataset is not None
        if self.sample_size:
            self.dataset = self.dataset.select(range(min(self.sample_size, len(self.dataset))))
        self._schema_validated = False  # Reset schema validation cache on load
        return self.dataset

    def validate_schema(self, sample_size: int | None = 10) -> bool:
        """Validate the dataset schema."""
        if self._schema_validated:
            return True
        if not self.dataset:
            self.load()
        assert self.dataset is not None
        # Use sample if sample_size is provided, else full dataset
        if sample_size is not None:
            indices = range(min(sample_size, len(self.dataset)))
            items = (self.dataset[i] for i in indices)
        else:
            items = self.dataset
        for item in items:
            DatasetItem(**item)
        self._schema_validated = True
        return True

    def get_dataset_name(self) -> str:
        """Get the dataset name."""
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        """Get the dataset info."""
        if not self.dataset:
            self.load()
        assert self.dataset is not None
        return self.dataset.info
