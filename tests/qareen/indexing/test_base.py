"""Vector store indexing contract tests."""

from __future__ import annotations

from abc import ABC

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.chroma_indexer import ChromaIndexer

REQUIRED_INDEXER_METHODS = frozenset({"index", "create_vectorstore", "get_embeddings"})


def test_vector_store_indexer_contract_and_naming() -> None:
    assert issubclass(VectorStoreIndexer, ABC)
    assert getattr(VectorStoreIndexer, "__abstractmethods__", set()) >= REQUIRED_INDEXER_METHODS

    class StubChromaIndexer(ChromaIndexer):
        def __init__(self) -> None:
            pass

        def index(
            self,
            alpha_values: list[float],
            batch_size: int = 100,
            sample_size: int | None = None,
        ) -> dict[float, VectorStore]:
            raise NotImplementedError()

        def create_vectorstore(
            self,
            dataset_name: str,
            model_id: str,
            alpha: float,
            environment: str = "dev",
        ) -> VectorStore:
            raise NotImplementedError()

        def get_embeddings(self) -> Embeddings:
            raise NotImplementedError()

    indexer = StubChromaIndexer()
    assert (
        indexer.get_collection_name(
            dataset_name="SQID Shots",
            environment="staging",
            model_id="google/siglip-base-patch16-224",
        )
        == "staging_sqid_shots_google_siglip-base-patch16-224"
    )
