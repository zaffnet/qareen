from abc import ABC, abstractmethod
from typing import Any
from qareen.dataset.schema import DatasetSchema

class VectorStoreIndexer(ABC):

    @abstractmethod
    def index(self, data: DatasetSchema, model: Any, alpha: float) -> None:
        """Takes dataset and creates vector store."""

    @abstractmethod
    def get_collection_name(self, dataset_name: str, model_id: str, alpha: float, environment: str) -> str:
        """Generates collection name from dataset_name, environment, model_id, and alpha."""

    @abstractmethod
    def create_vectorstore(self, collection_name: str) -> Any:
        """Creates LangChain VectorStore instance."""

    @abstractmethod
    def get_embeddings(self, model: Any) -> Any:
        """Returns LangChain Embeddings instance."""
