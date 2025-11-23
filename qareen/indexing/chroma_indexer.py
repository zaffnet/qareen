"""ChromaDB vector store indexer implementation."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from datasets import DatasetDict
from langchain_chroma import Chroma
from tqdm import tqdm

from qareen.indexing.base import VectorStoreIndexer
from qareen.models import Settings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

    from qareen.dataset.base import DatasetLoader

from qareen.indexing.embedding_model import EmbeddingModel, EmbeddingModelWrapper
from qareen.utils.image_utils import load_image
from qareen.utils.naming import get_collection_name


class ChromaIndexer(VectorStoreIndexer):
    """ChromaDB vector store indexer.

    Handles indexing datasets into ChromaDB with pre-computed multimodal embeddings.
    Supports multiple alpha values and environments.
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ChromaDB indexer.

        Args:
            dataset_loader: Dataset loader instance
            embedding_model: Embedding model instance
            settings: Configuration settings (uses defaults if not provided)
        """
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
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

    def __enter__(self) -> ChromaIndexer:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and cleanup resources."""
        self.close()

    def index(
        self,
        alpha_values: list[float],
        *,
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
        environment: str | None = None,
    ) -> dict[float, VectorStore]:
        """Create vector store indexes for multiple alpha values.

        Pre-computes embeddings for each item and creates separate collections
        for each alpha value.

        Args:
            alpha_values: List of alpha values to index
            rebuild: If True, deletes existing collections before indexing (expensive).
            batch_size: Batch size for processing
            sample_size: Optional sample size (overrides settings)
            environment: Environment (dev/staging/prod), defaults to settings.environment

        Returns:
            Dictionary mapping alpha values to VectorStore instances
        """
        dataset = self.dataset_loader.load()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = environment or self.settings.environment

        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                first_split = next(iter(dataset.keys()))
                dataset = dataset[first_split]

        if sample_size is not None:
            limit = sample_size
        elif environment == "dev":
            limit = self.settings.dev_sample_size
        else:
            limit = None

        dataset_len = len(dataset)
        if limit is not None:
            try:
                selected = dataset.select(range(min(limit, dataset_len)))
                if len(selected) > 0:
                    dataset = selected
                    dataset_len = len(dataset)
            except (AttributeError, TypeError):
                pass

        self.embedding_model.load_model()

        vectorstores: dict[float, VectorStore] = {}
        chroma_client = self._get_chroma_client()

        for alpha in alpha_values:
            collection_name = get_collection_name(
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
            )

            if rebuild:
                with contextlib.suppress(ValueError, NotFoundError):
                    chroma_client.delete_collection(name=collection_name)

            collection_metadata = {"hnsw:space": "cosine"}
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=EmbeddingModelWrapper(self.embedding_model),
                collection_metadata=collection_metadata,
            )

            for idx in tqdm(
                range(0, dataset_len, batch_size),
                desc=f"[Model: {model_id}] [Alpha: {alpha:.3f}]",
            ):
                batch = dataset[idx : idx + batch_size]

                if not isinstance(batch, dict):
                    batch = {col: batch[col] for col in batch.column_names}

                batch_size_actual = len(batch["text"])

                batch_documents = []
                batch_embeddings = []
                batch_metadatas = []
                batch_ids = []

                for i in range(batch_size_actual):
                    text = batch["text"][i]
                    image = batch["image"][i]

                    try:
                        image = load_image(image)

                        embedding = self.embedding_model.embed_multimodal(
                            image=image,
                            text=text,
                            alpha=alpha,
                        )

                        if not hasattr(embedding, "tolist"):
                            raise TypeError(
                                f"Embedding must have tolist() method, got {type(embedding)}"
                            )

                        doc_text = text if text is not None else f"[image-only sample {idx + i}]"
                        batch_documents.append(doc_text)
                        batch_embeddings.append(embedding.tolist())
                        batch_metadatas.append(
                            {
                                "alpha": alpha,
                                "index": idx + i,
                                "has_text": text is not None,
                                "has_image": image is not None,
                            },
                        )
                        batch_ids.append(f"{idx + i}")
                    except Exception as e:
                        raise RuntimeError(f"Embedding failed for item {idx + i}") from e

                vectorstore.add_texts(
                    texts=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )

            vectorstores[alpha] = vectorstore

        return vectorstores
