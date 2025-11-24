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
        """
        Initialize the HuggingFaceDatasetLoader with the target dataset, split, and any load-time options.
        
        Parameters:
            dataset_name (str): HuggingFace Datasets identifier to load.
            split (str): Dataset split to load (for example, "train" or "test").
            load_kwargs (object): Additional keyword arguments forwarded to `load_dataset`.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.load_kwargs = load_kwargs
        self._dataset: HFDataset | DatasetDict | None = None

    def load(self) -> HFDataset | DatasetDict:
        """
        Load and cache the HuggingFace dataset associated with this loader.
        
        Returns:
            HFDataset | DatasetDict: The loaded dataset object; a `Dataset` for single-split datasets or a `DatasetDict` for multi-split datasets.
        """
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name, split=self.split, **self.load_kwargs)
        return self._dataset

    def validate_schema(self) -> None:
        """
        Validate the loaded HuggingFace dataset's schema.
        
        Calls the shared schema validator for the dataset returned by self.load(). Raises an error if required fields are missing or the schema is invalid.
        """
        validate_dataset_schema(self.load(), MISSING_FIELDS_ERROR)

    def get_dataset_name(self) -> str:
        """
        Return the configured Hugging Face dataset identifier.
        
        Returns:
            dataset_name (str): The dataset name provided at initialization.
        """
        return self.dataset_name

    def get_dataset_info(self) -> dict[str, Any]:
        """
        Provide metadata for the configured HuggingFace dataset.
        
        Retrieves information for the dataset identified by the loader's dataset_name and split, including configuration, features, and split statistics.
        
        Returns:
            metadata (dict[str, Any]): Mapping of metadata fields (for example: configuration, features, and split sizes).
        """
        return extract_dataset_info(self.load(), self.dataset_name, self.split)