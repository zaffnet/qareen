"""Vector store indexing contract tests."""

from __future__ import annotations

from abc import ABC
from typing import Protocol

from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.chroma_indexer import ChromaIndexer


class Embeddings(Protocol):
    """Placeholder interface for embedding providers."""


class VectorStore(Protocol):
    """Placeholder interface for vector stores."""


REQUIRED_INDEXER_METHODS = frozenset({"index"})


def test_vector_store_indexer_contract_and_naming() -> None:
    """
    Verify the VectorStoreIndexer abstract contract and the collection naming convention.
    
    Asserts that VectorStoreIndexer is an abstract base class and that its abstract methods include the required indexer methods. Defines a minimal StubChromaIndexer with the expected method signatures used by the indexing subsystem. Also verifies that get_collection_name produces the expected collection identifier for the given dataset, environment, and model_id.
    """
    assert issubclass(VectorStoreIndexer, ABC)
    assert getattr(VectorStoreIndexer, "__abstractmethods__", set()) >= REQUIRED_INDEXER_METHODS

    class StubChromaIndexer(ChromaIndexer):
        def __init__(self) -> None:
            pass

        def index(
            self,
            alpha_values: list[float],
            *,
            rebuild: bool,
            batch_size: int = 100,
            sample_size: int | None = None,
            environment: str | None = None,
        ) -> dict[float, VectorStore]:
            """
            Index documents into vector stores for the given alpha values.
            
            Parameters:
                alpha_values (list[float]): Alpha values that determine how embeddings or indexing variants are produced.
                rebuild (bool): If True, recreate existing vector stores instead of updating them.
                batch_size (int): Number of items to process per indexing batch.
                sample_size (int | None): If provided, limit indexing to this many samples per alpha.
                environment (str | None): Environment name to scope or locate the target vector stores (e.g., "dev", "staging").
            
            Returns:
                dict[float, VectorStore]: A mapping from each alpha value to its resulting VectorStore.
            """
            raise NotImplementedError()

        def get_vectorstore(
            self,
            dataset_name: str,
            model_id: str,
            alpha: float,
            environment: str = "dev",
        ) -> VectorStore:
            """
            Return a VectorStore for the specified dataset, model, alpha, and environment.
            
            Parameters:
                dataset_name (str): Dataset name used to locate or name the collection.
                model_id (str): Model identifier used to locate or name the collection.
                alpha (float): Alpha value associated with the vectorstore (used to distinguish configurations).
                environment (str): Deployment environment identifier (default "dev").
            
            Returns:
                VectorStore: The vector store instance corresponding to the provided identifiers.
            """
            raise NotImplementedError()

        def get_embeddings(self) -> Embeddings:
            """
            Provide the embeddings provider used by the indexer.
            
            Returns:
                Embeddings: An embeddings provider instance used to produce vector embeddings.
            
            Raises:
                NotImplementedError: If the concrete indexer does not implement this method.
            """
            raise NotImplementedError()

    from qareen.utils.naming import get_collection_name

    assert (
        get_collection_name(
            dataset_name="Conceptual Captions",
            environment="staging",
            model_id="google/siglip-base-patch16-224",
        )
        == "staging_conceptual_captions_google_siglip_base_patch16_224"
    )