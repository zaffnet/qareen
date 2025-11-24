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
        """
        Initialize the ChromaRetriever with an embedding model and optional settings.
        
        Parameters:
            embedding_model (EmbeddingModel): Model used to produce embeddings for multimodal queries.
            settings (Settings | None): Configuration and paths for the retriever; if None, default Settings are created and their required directories are ensured.
        """
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.embedding_model = embedding_model
        self._chroma_client: Any = None

    def _get_chroma_client(self) -> Any:
        """
        Lazily creates and caches a Chroma client using the configured chroma database directory.
        
        Returns:
            The cached or newly created Chroma client instance.
        """
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        """
        Close the underlying Chroma client and clear the cached client reference.
        
        Releases resources held by the internal Chromadb client and sets the retriever's cached client to None.
        """
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __del__(self) -> None:
        """
        Ensure resources are released by closing the underlying Chroma client when the retriever is garbage-collected, suppressing any exceptions raised during cleanup.
        """
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaRetriever:
        """
        Enter the context manager and provide the retriever instance.
        
        Returns:
            ChromaRetriever: The same retriever instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """
        Release resources when exiting a context manager.
        
        Calls close() to ensure the underlying Chroma client and related resources are released.
        """
        self.close()

    def get_vectorstore(
        self, dataset_name: str, model_id: str, alpha: float, environment: str = "dev"
    ) -> Any:
        """
        Retrieve the Chroma collection for the specified dataset, model, alpha, and environment.
        
        Parameters:
            dataset_name (str): Dataset identifier used when the collection was created.
            model_id (str): Embedding/model identifier used when the collection was created.
            alpha (float): Fusion weight (alpha) used when the collection was indexed.
            environment (str): Deployment environment or namespace (e.g., "dev", "prod").
        
        Returns:
            Any: The Chroma collection object corresponding to the requested parameters.
        
        Raises:
            ValueError: If the named collection does not exist.
        """
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
        """
        Retrieve nearest-neighbor documents from the given vector store using a multimodal embedding computed from the provided image and/or text.
        
        Parameters:
            vectorstore (Any): Chroma collection-like object to query.
            image (PIL.Image.Image | str | None): Image or image path to include in the multimodal embedding; may be None.
            text (str | None): Text to include in the multimodal embedding; may be None.
            alpha (float): Mixing weight between image and text embeddings; must be between 0.0 and 1.0.
            k (int): Maximum number of results to return (after filtering).
            score_threshold (float | None): If provided, only include results with similarity >= this value.
        
        Returns:
            list[tuple[Document, float]]: A list of tuples where each tuple contains a Document and its similarity score in the range [0.0, 1.0].
        
        Raises:
            ValueError: If alpha is outside [0.0, 1.0], if the collection does not use the 'cosine' distance metric, or if the query alpha is inconsistent with the collection's indexed alpha.
        """
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
        for _id, metadata, doc_text, distance in zip(ids, metadatas, docs, distances, strict=True):
            similarity = max(0.0, min(1.0, 1.0 - (abs(distance) / 2.0)))
            skip = similarity > IDENTICAL_THRESHOLD
            if score_threshold is not None:
                skip = skip or similarity < score_threshold
            if skip:
                continue
            documents.append((Document(page_content=doc_text, metadata=metadata), similarity))
            if len(documents) >= k:
                break

        return documents

    def list_available_alphas(
        self, dataset_name: str, model_id: str, environment: str = "dev"
    ) -> list[float]:
        """
        List available alpha values for collections matching a dataset and model.
        
        Parameters:
            dataset_name (str): Name of the dataset.
            model_id (str): Identifier of the model used for the collection naming.
            environment (str): Environment suffix used in collection names (e.g., "dev", "prod").
        
        Returns:
            alphas (list[float]): Sorted list of alpha values parsed from matching collection names.
        """
        prefix = get_collection_name(dataset_name, model_id, None, environment)
        alphas = [
            float(match.group(1))
            for collection in self._get_chroma_client().list_collections()
            if collection.name.startswith(prefix)
            and (match := re.search(r"_a(\d+\.\d+)", collection.name))
        ]
        return sorted(alphas)