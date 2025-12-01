from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict

from qareen.models import DatasetItem

MISSING_FIELDS_ERROR = (
    "Dataset missing required fields: {missing_fields}. Available fields: {available_fields}"
)


def validate_dataset_schema(dataset: HFDataset | DatasetDict, error_template: str) -> None:
    if isinstance(dataset, dict):
        if not dataset:
            raise ValueError("DatasetDict has no splits")
        features = next(iter(dataset.values())).features
        non_empty_split = next((s for s in dataset.values() if len(s) > 0), None)
        if non_empty_split is None:
            raise ValueError("Dataset contains no non-empty splits")
        sample = non_empty_split[0]
    else:
        features = dataset.features
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        sample = dataset[0]

    missing_fields = {"text", "image"} - set(features.keys())
    if missing_fields:
        raise ValueError(
            error_template.format(
                missing_fields=missing_fields, available_fields=set(features.keys())
            )
        )

    DatasetItem(**sample)


def extract_dataset_info(
    dataset: HFDataset | DatasetDict, dataset_name: str, split: str | None = None
) -> dict[str, Any]:
    if isinstance(dataset, dict):
        features = list(next(iter(dataset.values())).features.keys()) if dataset else []
        return {
            "dataset_name": dataset_name,
            "splits": list(dataset.keys()),
            "num_rows": {k: len(v) for k, v in dataset.items()},
            "features": features,
        }
    info = {
        "dataset_name": dataset_name,
        "num_rows": len(dataset),
        "features": list(dataset.features.keys()),
    }
    if split is not None:
        info["split"] = split
    return info


class DatasetLoader(ABC):
    """Abstract base class for loading datasets in standardized format."""

    @abstractmethod
    def load(self) -> Any:
        """Load dataset and return in a standardized format (usually HF Dataset)."""
        pass

    @abstractmethod
    def validate_schema(self) -> None:
        """Validate that the dataset has the required fields (text, image)."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset identifier."""
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata about the dataset."""
        pass
