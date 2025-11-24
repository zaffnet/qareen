from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from chromadb.errors import NotFoundError
from datasets import DatasetDict
from tqdm import tqdm

from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.embedding_model import EmbeddingModel
from qareen.models import Settings
from qareen.utils.chroma_client import close_chroma_client, create_chroma_client
from qareen.utils.image_utils import load_image
from qareen.utils.naming import get_collection_name

if TYPE_CHECKING:
    from qareen.dataset.base import DatasetLoader


class ChromaIndexer(VectorStoreIndexer):
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        """
        Initializes a ChromaIndexer with a dataset loader, embedding model, and optional settings.
        
        Parameters:
            dataset_loader (DatasetLoader): Loader that provides the dataset to index.
            embedding_model (EmbeddingModel): Model used to compute multimodal embeddings.
            settings (Settings | None): Optional configuration; when omitted, defaults are created. The settings' directories are ensured to exist as a side effect.
        """
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model
        self._chroma_client = None

    def _get_chroma_client(self):
        """
        Lazily create and return the Chromadb client used by this indexer.
        
        Returns:
            The Chromadb client instance; if no client exists one will be created and cached.
        """
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        """
        Close the internal Chroma client and clear the cached client handle.
        
        This releases any resources held by the indexer's Chromadb client and sets the internal client reference to None.
        """
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __del__(self) -> None:
        """
        Ensure resources are released when the object is garbage-collected.
        
        Calls `close()` to release underlying resources and suppresses any exception raised during cleanup.
        """
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaIndexer:
        """
        Enter a context manager and provide this ChromaIndexer instance for use within the context.
        
        Returns:
            self (ChromaIndexer): The indexer instance to be used as the context-managed object.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """
        Ensure resources are released when exiting a context by closing the Chromadb client if it is open.
        
        Any exception information provided by the context manager protocol is ignored.
        """
        self.close()

    def index(
        self,
        alpha_values: list[float],
        *,
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
        environment: str | None = None,
    ) -> dict[float, Any]:
        """
        Index the dataset into Chroma collections for each provided alpha value.
        
        This creates (or, when `rebuild` is True, recreates) one Chroma collection per alpha and adds batches of multimodal embeddings computed from dataset text and images. The dataset may be sampled via `sample_size` or the configured dev sample size when `environment` is "dev". Each stored entry includes the document text (or an image-only placeholder), the embedding vector, and metadata containing the alpha, original index, and flags indicating presence of text and image.
        
        Parameters:
            alpha_values (list[float]): Alpha values to use when computing multimodal embeddings; a separate collection is created per alpha.
            rebuild (bool): If True, delete any existing collection with the same name before creating a new one.
            batch_size (int): Number of samples to process per batch when adding to collections.
            sample_size (int | None): Optional explicit maximum number of samples to index. If None and `environment` is "dev", the configured dev sample size is used.
            environment (str | None): Evaluation environment name used when naming collections; defaults to the indexer's configured environment when None.
        
        Returns:
            dict[float, Any]: Mapping from each alpha value to its corresponding Chroma collection object.
        
        Raises:
            TypeError: If an embedding object returned by the model does not implement a `tolist()` method.
        """
        dataset = self.dataset_loader.load()
        if isinstance(dataset, DatasetDict):
            dataset = dataset.get("train", next(iter(dataset.values())))

        env = environment or self.settings.environment
        limit = (
            sample_size
            if sample_size is not None
            else (self.settings.dev_sample_size if env == "dev" else None)
        )
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        self.embedding_model.load_model()
        client = self._get_chroma_client()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        vectorstores: dict[float, Any] = {}

        for alpha in alpha_values:
            name = get_collection_name(dataset_name, model_id, alpha, env)
            if rebuild:
                with contextlib.suppress(ValueError, NotFoundError):
                    client.delete_collection(name=name)

            collection = client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            dataset_len = len(dataset)

            for idx in tqdm(
                range(0, dataset_len, batch_size), desc=f"[Model: {model_id}] [Alpha: {alpha:.3f}]"
            ):
                batch = dataset[idx : idx + batch_size]
                batch_dict = (
                    batch
                    if isinstance(batch, dict)
                    else {col: batch[col] for col in batch.column_names}
                )

                docs, embeddings, metadatas, ids = [], [], [], []
                for i, (text, image) in enumerate(
                    zip(batch_dict["text"], batch_dict["image"], strict=True)
                ):
                    img = load_image(image)
                    emb = self.embedding_model.embed_multimodal(image=img, text=text, alpha=alpha)
                    if not hasattr(emb, "tolist"):
                        raise TypeError(f"Embedding must have tolist() method, got {type(emb)}")

                    docs.append(text or f"[image-only sample {idx + i}]")
                    embeddings.append(emb.tolist())
                    metadatas.append(
                        {
                            "alpha": alpha,
                            "index": idx + i,
                            "has_text": text is not None,
                            "has_image": img is not None,
                        }
                    )
                    ids.append(f"{idx + i}")

                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

            vectorstores[alpha] = collection

        return vectorstores