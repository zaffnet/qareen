from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from chromadb import Collection, PersistentClient
from chromadb.errors import NotFoundError

if TYPE_CHECKING:
    from PIL import Image

    from qareen.indexing.embedding_model import EmbeddingModel

from qareen.models import Settings
from qareen.utils.chroma_client import (
    close_chroma_client,
    create_chroma_client,
)
from qareen.utils.image_utils import load_image
from qareen.utils.naming import ALPHA_SUFFIX_PATTERN, get_collection_name

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
        self._chroma_client: PersistentClient | None = None

    def _get_chroma_client(self) -> PersistentClient:
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __enter__(self) -> ChromaRetriever:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_vectorstore(
        self, dataset_name: str, model_id: str, alpha: float, environment: str = "dev"
    ) -> Collection:
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
        vectorstore: Collection,
        image: Image.Image | str | None,
        text: str | None,
        alpha: float,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        metadata = getattr(vectorstore, "metadata", None)
        if metadata is None:
            # Assume missing metadata implies improper initialization or legacy collection
            # But if we want to be strict as requested:
            # "Detect the missing-metadata case explicitly and raise a clear error"
            raise ValueError("Collection has no metadata. Cannot verify distance metric.")

        distance_metric = metadata.get("hnsw:space")
        if distance_metric is None:
            raise ValueError(
                "Collection metadata missing 'hnsw:space'. Likely using default 'l2', "
                "but 'cosine' is required. Re-index with cosine distance."
            )

        if distance_metric != "cosine":
            raise ValueError(
                f"Collection uses '{distance_metric}' distance metric, "
                f"but qareen requires 'cosine'. Re-index with cosine distance."
            )

        sample = vectorstore.get(limit=1, include=["metadatas"])
        if sample.get("ids") and sample.get("metadatas"):
            metadatas = sample["metadatas"]
            # Handle potential nested list if get() returns nested list (safety check)
            first_meta = metadatas[0] if metadatas else {}
            if isinstance(first_meta, list):
                first_meta = first_meta[0] if first_meta else {}

            sample_alpha = first_meta.get("alpha")
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

        ids = (results.get("ids") or [[]])[0]
        if not ids:
            return []

        count = len(ids)
        # Ensure lists are of equal length to avoid zip issues
        metadatas = (results.get("metadatas") or [[]])[0]
        if not metadatas:
            metadatas = cast(list[dict[str, Any]], [{}] * count)

        docs = (results.get("documents") or [[]])[0]
        if not docs:
            docs = [""] * count

        distances = (results.get("distances") or [[]])[0]
        if not distances:
            distances = [0.0] * count

        documents = []
        skipped_identical = False
        # Remove strict=True to prevent crashes on slight API mismatch
        for _id, metadata, doc_text, distance in zip(ids, metadatas, docs, distances, strict=False):
            # Map Cosine Distance [0, 2] to Similarity [0, 1]
            # 0 (identical) -> 1.0
            # 1 (orthogonal) -> 0.5
            # 2 (opposite) -> 0.0
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
        # Generate prefix by stripping alpha suffix from a dummy name
        dummy_name = get_collection_name(dataset_name, model_id, 0.0, environment)
        prefix = re.sub(ALPHA_SUFFIX_PATTERN, "", dummy_name)

        collections = self._get_chroma_client().list_collections()
        if not collections:
            return []
        alphas = [
            float(match.group(1).replace("_", "."))
            for collection in collections
            if collection.name.startswith(prefix)
            and (match := ALPHA_SUFFIX_PATTERN.search(collection.name))
        ]
        return sorted(alphas)
