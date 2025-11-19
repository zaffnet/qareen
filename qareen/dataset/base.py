"""Abstract base class for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict

from qareen.dataset.schema import DatasetSchema

DATASETDICT_NO_SPLITS_ERROR = "DatasetDict has no splits"
NO_NON_EMPTY_SPLITS_ERROR = "Dataset contains no non-empty splits"
EMPTY_DATASET_ERROR = "Dataset is empty"


def validate_dataset_schema(
    dataset: HFDataset | DatasetDict,
    error_template: str,
) -> None:
    """Validate dataset has required text and image fields.

    Args:
        dataset: HuggingFace dataset or dataset dict to validate
        error_template: Error message template with {missing_fields} and {available_fields}

    Raises:
        ValueError: If required fields are missing or dataset is empty
    """
    if isinstance(dataset, dict):
        if not dataset:
            raise ValueError(DATASETDICT_NO_SPLITS_ERROR)
        features = next(iter(dataset.values())).features
    else:
        features = dataset.features

    required_fields = {"text", "image"}
    missing_fields = required_fields - set(features.keys())

    if missing_fields:
        raise ValueError(
            error_template.format(
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
            raise ValueError(NO_NON_EMPTY_SPLITS_ERROR)
        sample = non_empty_split[0]
    else:
        if len(dataset) > 0:
            sample = dataset[0]
        else:
            raise ValueError(EMPTY_DATASET_ERROR)

    DatasetSchema(**sample)


class DatasetLoader(ABC):
    """Abstract base class for loading datasets in standardized format.

    Implementations must provide methods to load, validate, and get metadata
    about datasets.
    """

    @abstractmethod
    def load(self) -> Any:
        """Load dataset and return in standardized format.

        Returns:
            Dataset in standardized format (implementation-specific)
        """
        pass

    @abstractmethod
    def validate_schema(self) -> None:
        """Validate dataset has required fields (text, image).

        Raises:
            ValueError: If required fields are missing
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return dataset identifier.

        Returns:
            Dataset name/identifier
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata.

        Returns:
            Dictionary containing dataset metadata (size, splits, etc.)
        """
        pass
