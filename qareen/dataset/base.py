from abc import ABC, abstractmethod
from typing import Any

from qareen.dataset.schema import DatasetSchema


class DatasetLoader(ABC):

    @abstractmethod
    def load(self) -> DatasetSchema:
        """Loads the dataset and returns it as a DatasetSchema object."""
        pass

    @abstractmethod
    def validate_schema(self, data: Any) -> bool:
        """Validates the dataset against the expected schema."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Extracts and returns the dataset identifier."""
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict:
        """Returns metadata about the dataset."""
        pass
