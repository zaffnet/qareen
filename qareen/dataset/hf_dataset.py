from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

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
        self.dataset: Dataset | None = None

    def load(self) -> Any:
        """Load the dataset from HuggingFace."""
        self.dataset = load_dataset(self.dataset_name, split="train")
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
        if not self.dataset:
            return False
        for item in self.dataset:
            DatasetSchema(**item)
        return True

    def get_dataset_name(self) -> str:
        """Get the dataset name."""
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        """Get the dataset info."""
        if not self.dataset:
            self.load()
        if not self.dataset:
            raise RuntimeError("Dataset failed to load; cannot retrieve info.")
        return dict(self.dataset.info)
