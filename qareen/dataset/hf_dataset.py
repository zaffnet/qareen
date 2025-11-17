from qareen.dataset.base import DatasetLoader
from qareen.dataset.schema import DatasetSchema, DatasetItem
from typing import Any
import datasets

class HuggingFaceDatasetLoader(DatasetLoader):

    def __init__(self, dataset_name: str, sample_size: int = -1):
        if sample_size != -1 and sample_size <= 0:
            raise ValueError("sample_size must be -1 (full dataset) or a positive integer")
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self._dataset = None

    def load(self) -> DatasetSchema:
        """Loads the dataset from HuggingFace and returns it as a DatasetSchema object."""
        self._dataset = datasets.load_dataset(self.dataset_name, split='train') # Default to train split

        if self.sample_size > 0:
            num_samples = min(self.sample_size, len(self._dataset))
            self._dataset = self._dataset.select(range(num_samples))

        items = [DatasetItem(text=row['text'], image=row['image']) for row in self._dataset]
        return DatasetSchema(dataset_name=self.dataset_name, data=items)

    def validate_schema(self, data: Any) -> bool:
        """Validates the dataset against the expected schema."""
        raise NotImplementedError

    def get_dataset_name(self) -> str:
        """Extracts and returns the dataset identifier."""
        return self.dataset_name

    def get_dataset_info(self) -> dict:
        """Returns metadata about the dataset."""
        raise NotImplementedError
