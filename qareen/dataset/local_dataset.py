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
    def __init__(self, dataset_path: str | Path) -> None:
        if not dataset_path or str(dataset_path).strip() == "":
            raise ValueError("Dataset path is empty")
        path = Path(dataset_path)
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        if not path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")
        self.dataset_path = path
        self._dataset: HFDataset | DatasetDict | None = None

    def load(self) -> HFDataset | DatasetDict:
        if self._dataset is None:
            self._dataset = load_from_disk(str(self.dataset_path))
        return self._dataset

    def validate_schema(self) -> None:
        validate_dataset_schema(self.load(), MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        return self.dataset_path.name

    def get_dataset_info(self) -> dict[str, Any]:
        return extract_dataset_info(self.load(), self.get_dataset_name())
