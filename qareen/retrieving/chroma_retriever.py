from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chromadb.errors import NotFoundError

if TYPE_CHECKING:
    from PIL import Image

    from qareen.indexing.embedding_model import EmbeddingModel

from qareen.models import Settings
from qareen.utils.chroma_client import close_chroma_client, create_chroma_client
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
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
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

        # We need to fetch more candidates for MMR, and we need their embeddings
        results = vectorstore.query(
            query_embeddings=[query_emb.tolist()],
            n_results=max(k, fetch_k) + 1,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        ids = (results.get("ids") or [[]])[0]
        if not ids:
            return []

        metadatas = (results.get("metadatas") or [[]])[0] or [{}] * len(ids)
        docs = (results.get("documents") or [[]])[0] or [""] * len(ids)
        distances = (results.get("distances") or [[]])[0] or [0.0] * len(ids)
        embeddings = (results.get("embeddings") or [[]])[0]

        if embeddings is None or len(embeddings) == 0:
            # Fallback if embeddings are not returned
            embeddings = [query_emb.tolist()] * len(ids)

        documents_with_scores = []
        skipped_identical = False
        valid_indices = []

        for idx, (distance, _id) in enumerate(zip(distances, ids, strict=True)):
            similarity = max(0.0, min(1.0, 1.0 - (abs(distance) / 2.0)))
            if similarity > IDENTICAL_THRESHOLD and not skipped_identical:
                skipped_identical = True
                continue
            if score_threshold is not None and similarity < score_threshold:
                continue
            documents_with_scores.append((idx, similarity))
            valid_indices.append(idx)

        if not documents_with_scores:
            return []

        # MMR logic
        selected_indices = []
        unselected_indices = valid_indices.copy()

        # Select first item (most similar to query)
        best_initial_idx = max(documents_with_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_initial_idx)
        unselected_indices.remove(best_initial_idx)

        # Convert to numpy arrays for fast similarity computation
        query_emb_np = np.array(query_emb)
        query_norm = np.linalg.norm(query_emb_np)
        if query_norm > 0:
            query_emb_np = query_emb_np / query_norm

        emb_np = np.array(embeddings)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_np = emb_np / norms

        while len(selected_indices) < min(k, len(valid_indices)):
            best_score = -float("inf")
            best_idx = -1

            for idx in unselected_indices:
                # Similarity to query (we could use the precomputed distance, but let's be exact)
                sim_to_query = max(0.0, min(1.0, 1.0 - (abs(distances[idx]) / 2.0)))

                # Max similarity to already selected
                selected_embs = emb_np[selected_indices]
                candidate_emb = emb_np[idx]

                # Cosine similarity is dot product of normalized vectors
                sims_to_selected = np.dot(selected_embs, candidate_emb)
                max_sim_to_selected = np.max(sims_to_selected)

                # MMR score
                mmr_score = mmr_lambda * sim_to_query - (1 - mmr_lambda) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
            else:
                break

        final_documents = []
        for idx in selected_indices:
            similarity = max(0.0, min(1.0, 1.0 - (abs(distances[idx]) / 2.0)))
            doc = Document(page_content=docs[idx], metadata=metadatas[idx])
            final_documents.append((doc, similarity))

        return final_documents

    def list_available_alphas(
        self, dataset_name: str, model_id: str, environment: str = "dev"
    ) -> list[float]:
        prefix = get_collection_name(dataset_name, model_id, None, environment)
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
