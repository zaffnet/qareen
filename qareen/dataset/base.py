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
    """
    Validate that a Hugging Face dataset contains required fields and a valid sample.
    
    Checks that the dataset (either a DatasetDict or a single dataset) is non-empty, that at least one split contains rows, and that the dataset features include the required "text" and "image" fields. If validation passes, attempts to construct a DatasetItem from the first available sample to ensure the sample conforms to the expected schema.
    
    Parameters:
        dataset (HFDataset | DatasetDict): The dataset or dataset dictionary to validate.
        error_template (str): A template string used to format an error message when required fields are missing.
            It will be called with `missing_fields` and `available_fields` named arguments.
    
    Raises:
        ValueError: If a DatasetDict has no splits, if no split contains rows, if a single dataset is empty,
                    or if required fields ("text", "image") are missing (formatted using `error_template`).
        TypeError or other exceptions raised by DatasetItem if the sample does not conform to the expected schema.
    """
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
    """
    Collect metadata for a Hugging Face dataset (HFDataset) or a DatasetDict.
    
    Parameters:
        dataset (HFDataset | DatasetDict): The dataset to inspect; may be a single dataset or a mapping of split name to datasets.
        dataset_name (str): Identifier to include in the returned metadata.
        split (str | None): Optional split name to include in the metadata for a single HFDataset.
    
    Returns:
        dict[str, Any]: Metadata dictionary containing:
            - "dataset_name" (str): The provided dataset_name.
            - For a DatasetDict:
                - "splits" (list[str]): List of split names.
                - "num_rows" (dict[str, int]): Mapping from split name to number of rows.
                - "features" (list[str]): Feature keys from the first split (empty list if DatasetDict is empty).
            - For a single HFDataset:
                - "num_rows" (int): Number of rows in the dataset.
                - "features" (list[str]): Feature keys of the dataset.
                - "split" (str) — present only if the `split` parameter was provided.
    """
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
    @abstractmethod
    def load(self) -> Any:
        """
        Load the dataset associated with this loader.
        
        Returns:
            Any: The loaded dataset object (for example, a `DatasetDict` or a single `Dataset`), ready for validation and consumption.
        """
        pass

    @abstractmethod
    def validate_schema(self) -> None:
        """
        Validate the loader's dataset structure and required fields.
        
        Checks that the dataset to be loaded is non-empty and contains the required fields "text" and "image".
        Raises a ValueError when the dataset is empty, when the dataset has no non-empty splits (for multi-split datasets),
        or when one or more required fields are missing (the message will indicate which fields are missing and which are present).
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Return the canonical name of the dataset targeted by this loader.
        
        Returns:
            dataset_name (str): The dataset identifier used by the loader (for example, a dataset id, repository name, or file path).
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """
        Provide metadata describing the loaded dataset.
        
        Returns:
            info (dict[str, Any]): Dictionary containing dataset metadata. Expected keys:
                - "dataset_name" (str): Name or identifier of the dataset.
                - "num_rows" (int | dict[str, int]): Total number of rows for a single-dataset loader, or a mapping of split names to row counts for dataset dicts.
                - "features" (list[str]): List of feature/column names present in the dataset.
                - "splits" (list[str], optional): List of split names when the dataset exposes multiple splits.
                - "split" (str, optional): The split name that was loaded when applicable.
        """
        pass