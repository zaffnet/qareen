"""ChromaDB retriever implementation."""

from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from PIL import Image

from qareen.indexing.embedding_model import EmbeddingModel, EmbeddingModelWrapper
from qareen.models import Settings
from qareen.utils.image_utils import load_image
from qareen.utils.naming import get_collection_name

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """ChromaDB retriever.

    Handles retrieving documents from ChromaDB using multimodal embeddings.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ChromaDB retriever.

        Args:
            embedding_model: Embedding model instance
            settings: Configuration settings (uses defaults if not provided)
        """
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.embedding_model = embedding_model
        self._chroma_client: chromadb.PersistentClient | None = None

    def _get_chroma_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.settings.chroma_db_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma_client

    def close(self) -> None:
        """Close ChromaDB client and release resources."""
        if hasattr(self, "_chroma_client") and self._chroma_client is not None:
            with contextlib.suppress(Exception):
                self._chroma_client.clear_system_cache()
            self._chroma_client = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaRetriever:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and cleanup resources."""
        self.close()

    def get_vectorstore(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float,
        environment: str = "dev",
    ) -> VectorStore:
        """Get VectorStore instance for existing collection.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value
            environment: Environment (dev/staging/prod)

        Returns:
            VectorStore instance
        """
        collection_name = get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            alpha=alpha,
            environment=environment,
        )

        chroma_client = self._get_chroma_client()

        try:
            chroma_client.get_collection(name=collection_name)
        except NotFoundError:
            raise ValueError(
                f"Collection '{collection_name}' does not exist for dataset '{dataset_name}', "
                f"model '{model_id}', alpha {alpha:.3f}, environment '{environment}'"
            ) from None

        collection_metadata = {"hnsw:space": "cosine"}
        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=EmbeddingModelWrapper(self.embedding_model),
            collection_metadata=collection_metadata,
        )

    def query_multimodal(
        self,
        vectorstore: VectorStore,
        image: Image.Image | str | None,
        text: str | None,
        alpha: float,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Any, float]]:
        """Query vectorstore with multimodal embedding.

        Performs similarity search using a multimodal query embedding combining
        image and text according to the specified alpha value.

        Args:
            vectorstore: VectorStore instance to query (must be a Chroma instance)
            image: Query image (PIL Image, URL string, local path, or None)
            text: Query text string or None
            alpha: Alpha value for weighting (0.0 = text-only, 1.0 = image-only)
            k: Number of similar results to return
            score_threshold: Optional minimum similarity score threshold (0.0-1.0)

        Returns:
            List of (Document, score) tuples sorted by similarity (higher is better)

        Raises:
            ValueError: If both image and text are None, or if alpha is invalid
            TypeError: If vectorstore is not a Chroma instance
        """
        if not isinstance(vectorstore, Chroma):
            raise TypeError(
                f"vectorstore must be a Chroma instance, got {type(vectorstore).__name__}",
            )

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        chroma_collection = vectorstore._collection
        sample_result = chroma_collection.get(limit=1, include=["metadatas"])

        if sample_result and sample_result["ids"] and len(sample_result["ids"]) > 0:
            sample_metadata = sample_result["metadatas"][0] if sample_result["metadatas"] else None
            if sample_metadata and "alpha" in sample_metadata:
                collection_alpha = float(sample_metadata["alpha"])
                if not abs(alpha - collection_alpha) < 1e-6:
                    raise ValueError(
                        f"Query alpha {alpha:.3f} does not match collection's indexed alpha "
                        f"{collection_alpha:.3f}"
                    )

        loaded_image = load_image(image)

        query_embedding = self.embedding_model.embed_multimodal(
            image=loaded_image,
            text=text,
            alpha=alpha,
        )

        query_embedding_list = query_embedding.tolist()

        results = chroma_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        documents = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            docs = results["documents"][0] if results["documents"] else [""] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for _doc_id, metadata, doc_text, distance in zip(
                ids,
                metadatas,
                docs,
                distances,
                strict=True,
            ):
                similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                if score_threshold is not None and similarity_score < score_threshold:
                    continue

                doc = Document(page_content=doc_text, metadata=metadata)
                documents.append((doc, similarity_score))

        return documents

    def list_available_alphas(
        self,
        dataset_name: str,
        model_id: str,
        environment: str = "dev",
    ) -> list[float]:
        """List available alpha values for a dataset/model combination.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            environment: Environment (dev/staging/prod)

        Returns:
            Sorted list of available alpha values
        """
        chroma_client = self._get_chroma_client()

        collections = chroma_client.list_collections()

        alphas = []
        prefix = get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            environment=environment,
        )

        for collection in collections:
            if collection.name.startswith(prefix):
                match = re.search(r"_a(\d+\.\d+)", collection.name)
                if match:
                    alphas.append(float(match.group(1)))

        return sorted(alphas)
