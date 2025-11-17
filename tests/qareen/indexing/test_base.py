"""Vector store indexing contract tests."""

from __future__ import annotations

from abc import ABC

import pytest

from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.exceptions import InvalidCollectionNameError

REQUIRED_INDEXER_METHODS = frozenset(
    {"index", "get_collection_name", "create_vectorstore", "get_embeddings"}
)


def test_vector_store_indexer_contract_and_naming() -> None:
    assert issubclass(VectorStoreIndexer, ABC)
    assert getattr(VectorStoreIndexer, "__abstractmethods__", set()) >= REQUIRED_INDEXER_METHODS

    class StubChromaIndexer(ChromaIndexer):
        def __init__(self) -> None:
            pass

        def index(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError()

        def create_vectorstore(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError()

        def get_embeddings(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError()

    indexer = StubChromaIndexer()
    with pytest.raises(InvalidCollectionNameError):
        indexer.get_collection_name(
            dataset_name="SQID Shots",
            environment="Staging",
            model_id="google/siglip-base-patch16-224",
            alpha=0.5,
        )
