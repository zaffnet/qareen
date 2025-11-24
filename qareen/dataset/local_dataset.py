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
        """
        Initialize the loader with a validated local dataset directory path.
        
        Parameters:
            dataset_path (str | Path): Path to a dataset directory on the local filesystem.
        
        Raises:
            ValueError: If `dataset_path` is empty or only whitespace, if the path does not exist, or if the path is not a directory.
        """
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
        """
        Load the dataset from the loader's configured path and cache it for subsequent calls.
        
        Returns:
            HFDataset | DatasetDict: The loaded Hugging Face dataset or a DatasetDict; if the dataset was previously loaded, the cached instance is returned.
        """
        if self._dataset is None:
            self._dataset = load_from_disk(str(self.dataset_path))
        return self._dataset

    def validate_schema(self) -> None:
        """
        Validate that the loaded dataset contains all required fields and conforms to the expected schema.
        """
        validate_dataset_schema(self.load(), MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        """
        Return the dataset directory's name.
        
        Returns:
            dataset_name (str): The name of the dataset directory.
        """
        return self.dataset_path.name

    def get_dataset_info(self) -> dict[str, Any]:
        """
        Retrieve metadata and descriptive information about the dataset represented by this loader.
        
        Returns:
            dict[str, Any]: A mapping containing dataset information such as name, features, splits, example counts, and other extracted metadata.
        """
        return extract_dataset_info(self.load(), self.get_dataset_name())