from abc import ABC
from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.chroma_indexer import ChromaIndexer

REQUIRED_INDEXER_METHODS = {"index", "create_vectorstore", "get_embeddings", "get_collection_name"}


def test_vector_store_indexer_is_abc() -> None:
    """Indexer base class must be an ABC to enforce contract."""
    assert issubclass(VectorStoreIndexer, ABC)


def test_vector_store_indexer_methods_are_abstract() -> None:
    """Indexer base class must define an abstract public API."""
    assert REQUIRED_INDEXER_METHODS <= getattr(VectorStoreIndexer, "__abstractmethods__", set())


def test_chroma_indexer_collection_naming() -> None:
    """ChromaDB indexer must produce clean, informative collection names."""
    stub_cls = type(
        "StubChromaIndexer",
        (ChromaIndexer,),
        {
            "__init__": lambda self: None,
            "index": lambda self, *a, **k: NotImplemented,
            "create_vectorstore": lambda self, *a, **k: NotImplemented,
            "get_embeddings": lambda self, *a, **k: NotImplemented,
        },
    )
    indexer = stub_cls()
    assert indexer.get_collection_name(
        dataset_name="SQID Shots",
        environment="Staging",
        model_id="google/siglip-base-patch16-224",
        alpha=0.5,
    ) == "staging_sqid_shots_google-siglip-base-patch16-224_0.5"
