from __future__ import annotations

import re
from typing import Any, List

from langchain_chroma import Chroma

from .base import VectorStoreIndexer
from .exceptions import InvalidCollectionNameError


class ChromaIndexer(VectorStoreIndexer):
    def index(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def get_collection_name(
        self,
        dataset_name: str,
        environment: str,
        model_id: str,
        alpha: float,
    ) -> str:
        name = f"{environment}_{dataset_name}_{model_id}_alpha{alpha:.2f}"
        sanitized_name = re.sub(r"[^a-z0-9_-]", "_", name.lower())
        sanitized_name = re.sub(r"_{2,}", "_", sanitized_name)
        if not re.match(r"^[a-z0-9_-]+$", sanitized_name):
            raise InvalidCollectionNameError(sanitized_name, [])
        return sanitized_name

    def create_vectorstore(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def get_embeddings(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def list_available_alphas(self) -> List[float]:
        raise NotImplementedError

    def validate_alpha_available(self, alpha: float) -> bool:
        raise NotImplementedError
