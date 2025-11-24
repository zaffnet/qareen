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
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model
        self._chroma_client = None

    def _get_chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaIndexer:
        return self

    def __exit__(self, *args: object) -> None:
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
