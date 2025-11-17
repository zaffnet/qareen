"""ChromaDB vector store indexer implementation."""

from __future__ import annotations

import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from PIL import Image
from tqdm import tqdm

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.models import EmbeddingModel


class DummyEmbeddings(Embeddings):
    """Dummy embeddings for pre-computed vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Not used - embeddings are pre-computed."""
        raise NotImplementedError("Embeddings are pre-computed")

    def embed_query(self, text: str) -> list[float]:
        """Not used - embeddings are pre-computed."""
        raise NotImplementedError("Embeddings are pre-computed")


class ChromaIndexer(VectorStoreIndexer):
    """ChromaDB vector store indexer.

    Handles indexing datasets into ChromaDB with pre-computed multimodal embeddings.
    Supports multiple alpha values and environments.

    Attributes:
        settings: Configuration settings
        dataset_loader: Dataset loader instance
        embedding_model: Embedding model instance
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
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model

    def index(
        self,
        alpha_values: list[float],
        batch_size: int = 100,
        sample_size: int | None = None,
    ) -> dict[float, VectorStore]:
        """Create vector store indexes for multiple alpha values.

        Pre-computes embeddings for each item and creates separate collections
        for each alpha value.

        Args:
            alpha_values: List of alpha values to index
            batch_size: Batch size for processing
            sample_size: Optional sample size (overrides settings)

        Returns:
            Dictionary mapping alpha values to VectorStore instances
        """
        dataset = self.dataset_loader.load()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = self.settings.environment

        if environment == "dev":
            limit = sample_size or self.settings.dev_sample_size
            dataset = dataset.select(range(min(limit, len(dataset))))

        self.embedding_model.load_model()

        vectorstores: dict[float, VectorStore] = {}
        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

        try:
            for alpha in alpha_values:
                collection_name = self.get_collection_name(
                    dataset_name=dataset_name,
                    model_id=model_id,
                    alpha=alpha,
                    environment=environment,
                )

            from contextlib import suppress

            with suppress(Exception):
                chroma_client.delete_collection(name=collection_name)

            documents = []
            embeddings_list = []
            metadatas = []
            ids = []

            for idx in tqdm(
                range(0, len(dataset), batch_size),
                desc=f"[Model: {model_id}] [Alpha: {alpha:.2f}]",
            ):
                batch = dataset[idx : idx + batch_size]

                if not isinstance(batch, dict):
                    batch = {col: batch[col] for col in batch.column_names}

                batch_size_actual = len(batch["text"])

                for i in range(batch_size_actual):
                    text = batch["text"][i]
                    image = batch["image"][i]

                    if isinstance(image, dict) and "bytes" in image:
                        from io import BytesIO

                        image = Image.open(BytesIO(image["bytes"]))
                    elif isinstance(image, str):
                        image = Image.open(image)

                    embedding = self.embedding_model.embed_multimodal(
                        image=image,
                        text=text,
                        alpha=alpha,
                    )

                    documents.append(text)
                    embeddings_list.append(embedding.tolist())
                    metadatas.append({"alpha": alpha, "index": idx + i})
                    ids.append(f"{idx + i}")

            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=DummyEmbeddings(),
            )

            vectorstore.add_texts(
                texts=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids,
            )

            vectorstores[alpha] = vectorstore
        finally:
            pass

        return vectorstores

    def create_vectorstore(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float,
        environment: str = "dev",
    ) -> VectorStore:
        """Create VectorStore instance for existing collection.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value
            environment: Environment (dev/staging/prod)

        Returns:
            VectorStore instance
        """
        collection_name = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            alpha=alpha,
            environment=environment,
        )

        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=DummyEmbeddings(),
        )

    def get_embeddings(self) -> Embeddings:
        """Return dummy embeddings instance.

        Returns:
            Dummy embeddings (actual embeddings are pre-computed)
        """
        return DummyEmbeddings()

    def get_collection_name(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float | None = None,
        environment: str = "dev",
    ) -> str:
        """Generate sanitized collection name using base class implementation.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value (optional)
            environment: Environment (dev/staging/prod)

        Returns:
            Sanitized collection name
        """
        return super().get_collection_name(dataset_name, model_id, alpha, environment)

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
        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

        collections = chroma_client.list_collections()

        alphas = []
        prefix = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            environment=environment,
        )

        for collection in collections:
            if collection.name.startswith(prefix):
                match = re.search(r"alpha(\d+\.\d+)", collection.name)
                if match:
                    alphas.append(float(match.group(1)))

        return sorted(alphas)
