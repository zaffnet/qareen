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
    def __init__(self, dataset_name: str, split: str = "train", **load_kwargs: object) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.load_kwargs = load_kwargs
        self._dataset: HFDataset | DatasetDict | None = None

    def load(self) -> HFDataset | DatasetDict:
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name, split=self.split, **self.load_kwargs)
        return self._dataset

    def validate_schema(self) -> None:
        validate_dataset_schema(self.load(), MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        return extract_dataset_info(self.load(), self.dataset_name, self.split)
