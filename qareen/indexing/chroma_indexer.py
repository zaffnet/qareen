"""ChromaDB implementation of :class:`VectorStoreIndexer`."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qareen.dataset.schema import DatasetItem

from .base import VectorStoreIndexer

_ALLOWED_CHARS_PATTERN = re.compile(r"[^a-z0-9_-]+")


class ChromaIndexer(VectorStoreIndexer):
    """Concrete indexer that structures collection naming for ChromaDB."""

    def __init__(
        self,
        *,
        environment: str = "dev",
        chroma_db_dir: str | Path = Path("chroma_db"),
        dev_sample_size: int | None = 1000,
    ) -> None:
        self.environment = environment
        self.chroma_db_dir = Path(chroma_db_dir)
        self.dev_sample_size = dev_sample_size

    def index(
        self,
        items: Sequence[DatasetItem],
        *,
        model_id: str,
        alpha: float,
    ) -> VectorStore:
        raise NotImplementedError("Concrete implementations must provide indexing logic.")

    def get_collection_name(
        self,
        dataset_name: str,
        environment: str,
        model_id: str,
        alpha: float | None = None,
    ) -> str:
        components = (
            self._sanitize(environment),
            self._sanitize(dataset_name),
            self._sanitize(model_id),
        )
        alpha_component = (self._format_alpha_component(alpha),) if alpha is not None else ()
        name = "_".join(part for part in (*components, *alpha_component) if part)
        return re.sub(r"_+", "_", name)

    def create_vectorstore(self, collection_name: str, *, model_id: str) -> VectorStore:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.get_embeddings(model_id),
            persist_directory=str(self.chroma_db_dir),
        )

    def get_embeddings(self, model_id: str) -> Embeddings:
        raise NotImplementedError("Embedding acquisition must be implemented by subclasses.")

    @staticmethod
    def _sanitize(value: str) -> str:
        normalized = value.strip().lower().replace("/", "_")
        normalized = _ALLOWED_CHARS_PATTERN.sub("_", normalized)
        return normalized.strip("_")

    @staticmethod
    def _format_alpha_component(alpha: float) -> str:
        formatted = f"alpha-{alpha:.3f}".rstrip("0").rstrip(".")
        return ChromaIndexer._sanitize(formatted)


__all__ = ["ChromaIndexer"]
