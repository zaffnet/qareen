from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class VectorStoreIndexer(ABC):
    @abstractmethod
    def index(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def get_collection_name(
        self,
        dataset_name: str,
        environment: str,
        model_id: str,
        alpha: float,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_vectorstore(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def list_available_alphas(self) -> List[float]:
        raise NotImplementedError

    def validate_alpha_available(self, alpha: float) -> bool:
        raise NotImplementedError
