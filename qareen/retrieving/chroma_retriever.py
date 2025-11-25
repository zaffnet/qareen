from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chromadb.errors import NotFoundError

if TYPE_CHECKING:
    from PIL import Image

    from qareen.indexing.embedding_model import EmbeddingModel

from qareen.models import Settings
from qareen.utils.chroma_client import close_chroma_client, create_chroma_client
from qareen.utils.image_utils import load_image
from qareen.utils.naming import get_collection_name

ALPHA_TOLERANCE = 1e-6
IDENTICAL_THRESHOLD = 0.999999


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]


class ChromaRetriever:
    def __init__(self, embedding_model: EmbeddingModel, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.embedding_model = embedding_model
        self._chroma_client: Any = None

    def _get_chroma_client(self) -> Any:
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaRetriever:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_vectorstore(
        self, dataset_name: str, model_id: str, alpha: float, environment: str = "dev"
    ) -> Any:
        name = get_collection_name(dataset_name, model_id, alpha, environment)
        try:
            return self._get_chroma_client().get_collection(name=name)
        except NotFoundError as e:
            msg = (
                f"Collection '{name}' does not exist for dataset '{dataset_name}', "
                f"model '{model_id}', alpha {alpha:.3f}, environment '{environment}'"
            )
            raise ValueError(msg) from e

    def query_multimodal(
        self,
        vectorstore: Any,
        image: Image.Image | str | None,
        text: str | None,
        alpha: float,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        metadata = getattr(vectorstore, "metadata", None) or {}
        distance_metric = metadata.get("hnsw:space", "l2")
        if distance_metric != "cosine":
            raise ValueError(
                f"Collection uses '{distance_metric}' distance metric, "
                f"but qareen requires 'cosine'. Re-index with cosine distance."
            )

        sample = vectorstore.get(limit=1, include=["metadatas"])
        if sample.get("ids") and sample.get("metadatas"):
            sample_alpha = sample["metadatas"][0].get("alpha")
            if sample_alpha is not None and abs(alpha - float(sample_alpha)) >= ALPHA_TOLERANCE:
                msg = (
                    f"Query alpha {alpha:.3f} does not match "
                    f"collection's indexed alpha {sample_alpha:.3f}"
                )
                raise ValueError(msg)

        loaded_img = load_image(image)
        query_emb = self.embedding_model.embed_multimodal(image=loaded_img, text=text, alpha=alpha)
        results = vectorstore.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k + 1,
            include=["metadatas", "documents", "distances"],
        )

        if not results.get("ids"):
            return []

        ids = results["ids"][0]
        metadatas_list = results.get("metadatas") or [[]]
        metadatas = metadatas_list[0] if metadatas_list else [{}] * len(ids)
        docs_list = results.get("documents") or [[]]
        docs = docs_list[0] if docs_list else [""] * len(ids)
        distances_list = results.get("distances") or [[]]
        distances = distances_list[0] if distances_list else [0.0] * len(ids)

        documents = []
        skipped_identical = False
        for _id, metadata, doc_text, distance in zip(ids, metadatas, docs, distances, strict=True):
            similarity = max(0.0, min(1.0, 1.0 - (abs(distance) / 2.0)))
            if similarity > IDENTICAL_THRESHOLD and not skipped_identical:
                skipped_identical = True
                continue
            if score_threshold is not None and similarity < score_threshold:
                continue
            documents.append((Document(page_content=doc_text, metadata=metadata), similarity))
            if len(documents) >= k:
                break

        return documents

    def list_available_alphas(
        self, dataset_name: str, model_id: str, environment: str = "dev"
    ) -> list[float]:
        prefix = get_collection_name(dataset_name, model_id, None, environment)
        alphas = [
            float(match.group(1))
            for collection in self._get_chroma_client().list_collections()
            if collection.name.startswith(prefix)
            and (match := re.search(r"_a(\d+\.\d+)", collection.name))
        ]
        return sorted(alphas)
