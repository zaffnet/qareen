from __future__ import annotations

from typing import Any

from datasets import load_dataset

from .base import DatasetLoader
from .schema import DatasetSchema


class HuggingFaceDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name: str, sample_size: int | None = None):
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.dataset = None

    def load(self) -> Any:
        self.dataset = load_dataset(self.dataset_name, split="train")
        assert self.dataset is not None
        if self.sample_size:
            self.dataset = self.dataset.select(range(self.sample_size))
        return self.dataset

    def validate_schema(self) -> bool:
        if not self.dataset:
            self.load()
        assert self.dataset is not None
        for item in self.dataset:
            DatasetSchema(**item)
        return True

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        if not self.dataset:
            self.load()
        assert self.dataset is not None
        return self.dataset.info
