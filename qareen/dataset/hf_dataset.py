"""HuggingFace datasets implementation for :class:`DatasetLoader`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

from .base import DatasetLoader
from .schema import DatasetItem, DatasetSchema


class HuggingFaceDatasetLoader(DatasetLoader):
    """Load datasets via the HuggingFace `datasets` library with schema validation."""

    def __init__(self, dataset_id: str, *, split: str | None = None, **load_kwargs: Any) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.load_kwargs = load_kwargs
        self._dataset: DatasetDict | IterableDataset | None = None

    def load(self) -> Sequence[DatasetItem]:
        dataset = load_dataset(self.dataset_id, split=self.split, **self.load_kwargs)
        self._dataset = dataset  # Cache for downstream metadata access

        raw_records: Iterable[dict[str, Any]]
        if isinstance(dataset, DatasetDict):
            raw_records = cast(
                Iterable[dict[str, Any]],
                (record for split_dataset in dataset.values() for record in split_dataset),
            )
        elif isinstance(dataset, (IterableDataset, Dataset)):
            raw_records = cast(Iterable[dict[str, Any]], dataset)
        else:
            raw_records = cast(Iterable[dict[str, Any]], dataset)
        return self.validate_schema(raw_records)

    def validate_schema(self, records: Iterable[dict[str, Any]]) -> Sequence[DatasetItem]:
        validated: list[DatasetItem] = []
        for record in records:
            schema = DatasetSchema(**record)
            validated.append(DatasetItem(**schema.model_dump()))
        return validated

    def get_dataset_name(self) -> str:
        if self._dataset is None:
            return self.dataset_id
        info = getattr(self._dataset, "info", None)
        if info and getattr(info, "config_name", None):
            return str(info.config_name)
        return self.dataset_id

    def get_dataset_info(self) -> dict[str, Any]:
        if self._dataset is None:
            return {"dataset_id": self.dataset_id, "split": self.split}
        info = getattr(self._dataset, "info", None)
        if info is None:
            return {"dataset_id": self.dataset_id, "split": self.split}
        return {
            "description": getattr(info, "description", None),
            "features": getattr(info, "features", None),
            "homepage": getattr(info, "homepage", None),
            "citation": getattr(info, "citation", None),
            "version": getattr(info, "version", None),
        }


__all__ = ["HuggingFaceDatasetLoader"]
