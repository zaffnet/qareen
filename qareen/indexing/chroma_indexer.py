from qareen.indexing.base import VectorStoreIndexer
from qareen.dataset.schema import DatasetSchema
from typing import Any
import re

class ChromaIndexer(VectorStoreIndexer):

    def __init__(self, db_path: str = "chroma_db/"):
        self.db_path = db_path

    def index(self, data: DatasetSchema, model: Any, alpha: float) -> None:
        """Takes dataset and creates vector store."""
        raise NotImplementedError

    def get_collection_name(self, dataset_name: str, model_id: str, alpha: float, environment: str) -> str:
        """Generates collection name from dataset_name, environment, model_id, and alpha."""
        sanitized_dataset_name = re.sub(r'\s+', '_', dataset_name).lower()
        sanitized_model_id = model_id.replace("/", "-")
        return f"{environment.lower()}_{sanitized_dataset_name}_{sanitized_model_id}_{alpha}"

    def create_vectorstore(self, collection_name: str) -> Any:
        """Creates LangChain VectorStore instance."""
        raise NotImplementedError

    def get_embeddings(self, model: Any) -> Any:
        """Returns LangChain Embeddings instance."""
        raise NotImplementedError
