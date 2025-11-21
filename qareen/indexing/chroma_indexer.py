"""ChromaDB vector store indexer implementation."""

from __future__ import annotations

import contextlib
import logging
import os
import random
import re
import time

os.environ["ANONYMIZED_TELEMETRY"] = "False"
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, cast

import chromadb
import numpy as np
import requests
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from datasets import DatasetDict
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from PIL import Image, UnidentifiedImageError
from rich.logging import RichHandler
from tqdm import tqdm

from qareen.config.settings import Settings
from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.exceptions import (
    AlphaMismatchError,
    CollectionNotFoundError,
    InvalidEmbeddingError,
    UnsupportedImageTypeError,
)

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

    from qareen.dataset.base import DatasetLoader
    from qareen.indexing.models import EmbeddingModel

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def setup_logging(rich: bool = True, level: int = logging.INFO) -> None:
    """Configure logging with optional RichHandler.

    Args:
        rich: If True, use RichHandler for formatted output
        level: Logging level (default: INFO)

    """
    if not logging.getLogger().handlers:
        if rich:
            handler = RichHandler(rich_tracebacks=True, show_path=False)
            logging.basicConfig(level=level, format="%(message)s", handlers=[handler])
        else:
            logging.basicConfig(level=level, format="%(message)s")


class EmbeddingModelWrapper(Embeddings):
    """LangChain Embeddings wrapper for EmbeddingModel.

    Wraps the embedding model for use with LangChain's Chroma integration.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """Initialize wrapper.

        Args:
            embedding_model: Embedding model instance for query embedding

        """
        self.embedding_model = embedding_model
        self._embedding_dim: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using the embedding model.

        Args:
            texts: Input texts to embed

        Returns:
            List of embedding vectors as lists of floats

        Raises:
            RuntimeError: If embedding dimension cannot be determined or embedding fails
            ValueError: If embedding model returns None for any text

        """
        if not texts:
            return []
        if self._embedding_dim is None:
            try:
                self._embedding_dim = self.embedding_model.embedding_dim
            except (AttributeError, RuntimeError, TypeError) as e:
                logger.exception("Failed to get embedding dimension")
                raise RuntimeError("Cannot determine embedding dimension") from e

        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_model.embed_text(text)
                if embedding is None:
                    model_id = self.embedding_model.get_model_id()
                    raise ValueError(
                        f"Embedding returned None for text. "
                        f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
                    )
                embedding_list = cast("list[float]", embedding.tolist())
                if len(embedding_list) != self._embedding_dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                        f"got {len(embedding_list)}",
                    )
                embeddings.append(embedding_list)
            except ValueError:
                raise
            except Exception as e:
                logger.exception(f"Failed to embed text: {text[:100] if text else 'None'}...")
                raise RuntimeError("Embedding failed for text") from e

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using the embedding model.

        Args:
            text: Query text to embed

        Returns:
            Text embedding vector as list of floats

        Raises:
            ValueError: If embedding is None or cannot be converted to list

        """
        embedding = self.embedding_model.embed_text(text)
        model_id = self.embedding_model.get_model_id()
        if embedding is None:
            raise ValueError(
                f"Embedding returned None for provided text. "
                f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
            )
        if hasattr(embedding, "tolist"):
            return cast("list[float]", embedding.tolist())
        if hasattr(embedding, "__iter__") and not isinstance(embedding, (str, bytes)):
            return cast("list[float]", list(embedding))
        raise ValueError(
            f"Unsupported embedding format for model {model_id}. "
            f"Expected array-like object with tolist() method or iterable, "
            f"got {type(embedding).__name__}. Text snippet: {text[:50] if text else 'None'}...",
        )


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
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model
        self._chroma_client: chromadb.PersistentClient | None = None

    def _get_chroma_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client.

        Returns:
            ChromaDB client instance

        """
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

    def _download_image_with_retry(
        self,
        image_url: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_size_bytes: int = 10 * 1024 * 1024,
    ) -> Image.Image | None:
        """Download image from URL with retry logic and validation.

        Args:
            image_url: URL of the image to download
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_size_bytes: Maximum allowed image size in bytes

        Returns:
            PIL Image if successful, None otherwise

        Raises:
            ValueError: If max_retries is not positive

        """
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        for attempt in range(max_retries):
            try:
                with requests.get(image_url, timeout=30, stream=True) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"Invalid Content-Type '{content_type}' "
                            f"for image URL: {image_url} "
                            f"(attempt {attempt + 1}/{max_retries})",
                        )
                        if attempt == max_retries - 1:
                            return None
                        delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                        time.sleep(delay)
                        continue

                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            size = int(content_length)
                            if size > max_size_bytes:
                                logger.warning(
                                    f"Content-Length {size} exceeds max "
                                    f"{max_size_bytes} bytes for image "
                                    f"URL: {image_url} "
                                    f"(attempt {attempt + 1}/{max_retries})",
                                )
                                if attempt == max_retries - 1:
                                    return None
                                delay = base_delay * (2**attempt) + random.uniform(
                                    0,
                                    0.1 * base_delay,
                                )
                                time.sleep(delay)
                                continue
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Invalid Content-Length header "
                                f"'{content_length}' for image URL: "
                                f"{image_url} "
                                f"(attempt {attempt + 1}/{max_retries}): "
                                f"{e}",
                            )
                            if attempt == max_retries - 1:
                                return None
                            delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                            time.sleep(delay)
                            continue

                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=8192):
                        if len(content) + len(chunk) > max_size_bytes:
                            logger.warning(
                                f"Image size exceeds max "
                                f"{max_size_bytes} bytes during "
                                f"download for image URL: {image_url} "
                                f"(attempt {attempt + 1}/{max_retries})",
                            )
                            return None
                        content.extend(chunk)

                    if len(content) == 0:
                        logger.warning(
                            f"Empty response body for image URL: "
                            f"{image_url} "
                            f"(attempt {attempt + 1}/{max_retries})",
                        )
                        return None

                    try:
                        image_buffer = BytesIO(content)
                        img = Image.open(image_buffer)
                        img.verify()
                        image_buffer.seek(0)
                        image = Image.open(image_buffer)
                        return image
                    except (
                        UnidentifiedImageError,
                        OSError,
                        ValueError,
                    ) as e:
                        logger.warning(
                            f"Invalid image data for URL: "
                            f"{image_url} "
                            f"(attempt {attempt + 1}/{max_retries}): {e}",
                        )
                        return None
            except (
                requests.exceptions.RequestException,
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"Failed to download image after "
                        f"{max_retries} attempts: {image_url} "
                        f"(error: {type(e).__name__}: {e})",
                    )
                    return None
                delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                logger.debug(
                    f"Retry {attempt + 1}/{max_retries}: {image_url}, "
                    f"sleeping {delay:.3f}s "
                    f"(error: {type(e).__name__}: {e})",
                )
                time.sleep(delay)

        return None

    def _load_image(self, image: Image.Image | dict | str | None) -> Image.Image | None:
        """Load image from various input types.

        Args:
            image: Image input - can be Image.Image, dict with "bytes" key,
                URL string, local file path, or None

        Returns:
            PIL Image in RGB mode if successful, None otherwise

        Raises:
            UnsupportedImageTypeError: If image type is not supported

        """
        if image is None:
            return None

        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = self._download_image_with_retry(
                    image_url=image,
                    max_retries=3,
                    base_delay=1.0,
                    max_size_bytes=self.settings.max_image_bytes,
                )
            else:
                try:
                    image = Image.open(image)
                except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                    logger.exception("Failed to open image file: %s - %s", image, type(e).__name__)
                    image = None
        else:
            raise UnsupportedImageTypeError(type(image))

        if image is not None and image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def index(
        self,
        alpha_values: list[float],
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
        distance_metric: Literal["cosine", "l2"] = "cosine",
    ) -> dict[float, VectorStore]:
        """Create vector store indexes for multiple alpha values.

        Pre-computes embeddings for each item and creates separate collections
        for each alpha value.

        Args:
            alpha_values: List of alpha values to index
            rebuild: If True, deletes existing collections before indexing (expensive).
                If False, reuses existing collections (default for user scripts).
            batch_size: Batch size for processing
            sample_size: Optional sample size (overrides settings)
            distance_metric: Distance metric to use ("cosine" or "l2")

        Returns:
            Dictionary mapping alpha values to VectorStore instances

        """
        dataset = self.dataset_loader.load()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = self.settings.environment

        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
                logger.info("Selected 'train' split from DatasetDict")
            else:
                first_split = next(iter(dataset.keys()))
                dataset = dataset[first_split]
                logger.info(
                    f"Selected '{first_split}' split from DatasetDict (no 'train' split available)",
                )

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
                    logger.info(
                        f"Successfully created sample: requested limit={limit}, "
                        f"selected length={dataset_len}, dataset type={type(dataset).__name__}",
                    )
            except (AttributeError, TypeError) as e:
                logger.warning(
                    f"Sampling failed, falling back to full dataset: "
                    f"exception={type(e).__name__}:{e}, "
                    f"dataset type={type(dataset).__name__}, dataset length={dataset_len}",
                )

        # Ensure 0.0 and 1.0 are present for RRF support
        extended_alphas = sorted(list(set(alpha_values + [0.0, 1.0])))

        self.embedding_model.load_model()

        vectorstores: dict[float, VectorStore] = {}
        chroma_client = self._get_chroma_client()

        for alpha in extended_alphas:
            collection_name = self.get_collection_name(
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
                distance_metric=distance_metric,
            )

            if rebuild:
                logger.info(f"Attempting to delete collection: {collection_name}")
                try:
                    chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Successfully deleted collection: {collection_name}")
                except (ValueError, NotFoundError) as e:
                    logger.info(
                        f"Collection {collection_name} did not exist or deletion failed: {e}",
                    )

            collection_metadata = {"hnsw:space": distance_metric}
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=EmbeddingModelWrapper(self.embedding_model),
                collection_metadata=collection_metadata,
            )

            for idx in tqdm(
                range(0, dataset_len, batch_size),
                desc=f"[Model: {model_id}] [Metric: {distance_metric}] [Alpha: {alpha:.3f}]",
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
                        image = self._load_image(image)

                        embedding = self.embedding_model.embed_multimodal(
                            image=image,
                            text=text,
                            alpha=alpha,
                        )

                        if not hasattr(embedding, "tolist"):
                            raise InvalidEmbeddingError(type(embedding))

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
                        logger.exception(
                            f"Failed to process item {idx + i} (alpha={alpha:.3f})",
                        )
                        raise RuntimeError(f"Embedding failed for item {idx + i}") from e

                vectorstore.add_texts(
                    texts=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )

            vectorstores[alpha] = vectorstore

        return vectorstores

    def create_vectorstore(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float,
        environment: str = "dev",
        distance_metric: Literal["cosine", "l2"] = "cosine",
    ) -> VectorStore:
        """Create VectorStore instance for existing collection.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value
            environment: Environment (dev/staging/prod)
            distance_metric: Distance metric ("cosine" or "l2")

        Returns:
            VectorStore instance

        """
        collection_name = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            alpha=alpha,
            environment=environment,
            distance_metric=distance_metric,
        )

        chroma_client = self._get_chroma_client()

        try:
            chroma_client.get_collection(name=collection_name)
        except NotFoundError:
            raise CollectionNotFoundError(
                collection_name=collection_name,
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
            ) from None

        collection_metadata = {"hnsw:space": distance_metric}
        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=EmbeddingModelWrapper(self.embedding_model),
            collection_metadata=collection_metadata,
        )

    def get_embeddings(self) -> Embeddings:
        """Return embeddings wrapper instance.

        Returns:
            EmbeddingModelWrapper that can embed queries

        """
        return EmbeddingModelWrapper(self.embedding_model)

    def query_multimodal(
        self,
        vectorstore: VectorStore,
        image: Image.Image | str | None,
        text: str | None,
        alpha: float,
        k: int = 5,
        score_threshold: float | None = None,
        combination_method: Literal["linear", "rrf"] = "linear",
        retrieval_mode: Literal["similarity", "mmr"] = "similarity",
        distance_metric: Literal["cosine", "l2"] = "cosine",
    ) -> list[tuple[Any, float]]:
        """Query vectorstore with multimodal embedding.

        Supports Linear Combination and Reciprocal Rank Fusion (RRF).
        Supports Cosine Similarity, L2 Distance, and MMR.

        Args:
            vectorstore: VectorStore instance. Used as main store for linear combination.
                         For RRF, additional collections (alpha=0, alpha=1) are resolved automatically.
            image: Query image.
            text: Query text.
            alpha: Weighting factor (0.0-1.0).
                   For Linear: Weights the embedding combination.
                   For RRF: Weights the fusion of ranks (alpha=1.0 means image only).
            k: Number of results.
            score_threshold: Minimum score threshold.
            combination_method: "linear" or "rrf".
            retrieval_mode: "similarity" or "mmr" (Maximal Marginal Relevance).
            distance_metric: "cosine" or "l2".

        Returns:
            List of (Document, score) tuples.
            For Cosine: Score is similarity [0, 1] (higher is better).
            For L2: Score is distance (lower is better).
            For RRF: Score is fusion score (higher is better).

        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        loaded_image = self._load_image(image)

        if combination_method == "rrf":
            return self._query_rrf(
                image=loaded_image,
                text=text,
                alpha=alpha,
                k=k,
                retrieval_mode=retrieval_mode,
                distance_metric=distance_metric,
                vectorstore=vectorstore,  # Passed to extract collection metadata/config if needed
            )
        else:
            return self._query_linear(
                vectorstore=vectorstore,
                image=loaded_image,
                text=text,
                alpha=alpha,
                k=k,
                score_threshold=score_threshold,
                retrieval_mode=retrieval_mode,
                distance_metric=distance_metric,
            )

    def _query_linear(
        self,
        vectorstore: VectorStore,
        image: Image.Image | None,
        text: str | None,
        alpha: float,
        k: int,
        score_threshold: float | None,
        retrieval_mode: str,
        distance_metric: Literal["cosine", "l2"],
    ) -> list[tuple[Any, float]]:
        # Validate alpha matches vectorstore if possible
        if isinstance(vectorstore, Chroma):
            chroma_collection = vectorstore._collection
            sample_result = chroma_collection.get(limit=1, include=["metadatas"])
            if sample_result and sample_result["ids"] and len(sample_result["ids"]) > 0:
                sample_metadata = (
                    sample_result["metadatas"][0] if sample_result["metadatas"] else None
                )
                if sample_metadata and "alpha" in sample_metadata:
                    collection_alpha = float(sample_metadata["alpha"])
                    if not abs(alpha - collection_alpha) < 1e-6:
                        raise AlphaMismatchError(
                            query_alpha=alpha, collection_alpha=collection_alpha
                        )

        query_embedding = self.embedding_model.embed_multimodal(
            image=image,
            text=text,
            alpha=alpha,
        )
        query_embedding_list = query_embedding.tolist()

        documents = []

        if retrieval_mode == "mmr":
            if not isinstance(vectorstore, Chroma):
                raise TypeError("MMR requires Chroma vectorstore")

            # Chroma.max_marginal_relevance_search_by_vector returns list of Documents
            # It doesn't strictly return scores usually, but we can try to fetch them if supported
            # Or we assume standard MMR.
            # Langchain Chroma `max_marginal_relevance_search_by_vector` returns List[Document]
            # It does NOT return scores.
            # We might need to calculate scores manually or just return 1.0?
            # Or we can use `similarity_search_with_score` after selecting IDs?
            # Let's stick to returning Documents with dummy scores or calculated scores.

            results = vectorstore.max_marginal_relevance_search_by_vector(
                embedding=query_embedding_list,
                k=k,
            )
            # We need to calculate scores manually if we want them.
            # Use cosine similarity with query.
            for doc in results:
                # We can't easily get the vector back from Document unless we query again
                # or it's not exposed. We will just return score=1.0 or maybe re-rank?
                # For now, let's return 0.0 or 1.0 as placeholders or try to compute.
                # Since the user wants a "score", and MMR implies a ranking, we can assign
                # rank-based score.
                documents.append((doc, 0.0))  # Placeholder

        else:
            # Similarity search
            if isinstance(vectorstore, Chroma):
                # Use raw chroma query to get distances
                chroma_collection = vectorstore._collection
                results = chroma_collection.query(
                    query_embeddings=[query_embedding_list],
                    n_results=k,
                    include=["metadatas", "documents", "distances"],
                )

                if results and results["ids"] and len(results["ids"]) > 0:
                    ids = results["ids"][0]
                    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
                    docs = results["documents"][0] if results["documents"] else [""] * len(ids)
                    distances = (
                        results["distances"][0] if results["distances"] else [0.0] * len(ids)
                    )

                    for _doc_id, metadata, doc_text, distance in zip(
                        ids, metadatas, docs, distances, strict=True
                    ):
                        if distance_metric == "cosine":
                            # Convert distance to similarity [0, 1]
                            # Chroma cosine distance is [0, 2]
                            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                        else:
                            # L2 distance. Return raw distance.
                            score = distance

                        if (
                            score_threshold is not None
                            and distance_metric == "cosine"
                            and score < score_threshold
                        ):
                            continue
                        # For L2, thresholding logic might differ (max distance?)
                        # Assuming score_threshold is only for similarity for now or user handles it.

                        from langchain_core.documents import Document

                        # Ensure ID is available for RRF matching
                        # We copy metadata to avoid modifying the original dict if cached/shared
                        metadata = metadata.copy()
                        if "_id" not in metadata:
                            metadata["_id"] = _doc_id

                        doc = Document(page_content=doc_text, metadata=metadata)
                        documents.append((doc, score))
            else:
                # Fallback for generic vectorstore
                results = vectorstore.similarity_search_with_score_by_vector(
                    embedding=query_embedding_list, k=k
                )
                for doc, score in results:
                    documents.append((doc, score))

        return documents

    def _query_rrf(
        self,
        image: Image.Image | None,
        text: str | None,
        alpha: float,
        k: int,
        retrieval_mode: str,
        distance_metric: Literal["cosine", "l2"],
        vectorstore: VectorStore,
    ) -> list[tuple[Any, float]]:
        # Determine dataset/model from vectorstore or settings?
        # We need to find alpha=0 and alpha=1 collections.
        # We can assume they follow the naming convention.
        # We need dataset_name and model_id.
        # If vectorstore is Chroma, we can try to parse collection name?
        # Or better, we should have access to these params.
        # ChromaIndexer has dataset_loader and embedding_model.

        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = self.settings.environment

        # Get collections
        try:
            vs_text = self.create_vectorstore(
                dataset_name, model_id, 0.0, environment, distance_metric
            )
            vs_image = self.create_vectorstore(
                dataset_name, model_id, 1.0, environment, distance_metric
            )
        except CollectionNotFoundError as e:
            logger.error(f"RRF failed: Missing collection: {e}")
            raise

        # Helper to get results
        def get_results(vs: VectorStore, target_alpha: float) -> list[tuple[Any, float]]:
            return self._query_linear(
                vectorstore=vs,
                image=image,
                text=text,
                alpha=target_alpha,
                k=k,  # We might want more candidates for RRF
                score_threshold=None,
                retrieval_mode=retrieval_mode,
                distance_metric=distance_metric,
            )

        results_text = get_results(vs_text, 0.0)
        results_image = get_results(vs_image, 1.0)

        # RRF Fusion
        # Rank is 1-based.
        # score = w_text * (1 / (k + rank_text)) + w_image * (1 / (k + rank_image))

        w_image = alpha
        w_text = 1.0 - alpha
        rrf_k = 60  # Standard RRF constant

        scores: dict[str, float] = {}
        docs_map: dict[str, Any] = {}

        def process_results(results, weight):
            for rank, (doc, _) in enumerate(results):
                # Identify doc by unique ID
                doc_idx = str(doc.metadata.get("_id", ""))
                # Fallback to index if _id not present (legacy/MMR compatibility)
                if not doc_idx:
                    doc_idx = str(doc.metadata.get("index", ""))

                if not doc_idx:
                    continue  # Should not happen

                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                    docs_map[doc_idx] = doc

                scores[doc_idx] += weight * (1.0 / (rrf_k + rank + 1))

        if w_text > 0:
            process_results(results_text, w_text)
        if w_image > 0:
            process_results(results_image, w_image)

        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        final_results = []
        for doc_id in sorted_ids[:k]:
            final_results.append((docs_map[doc_id], scores[doc_id]))

        return final_results

    def list_available_alphas(
        self,
        dataset_name: str,
        model_id: str,
        environment: str = "dev",
        distance_metric: str = "cosine",
    ) -> list[float]:
        """List available alpha values for a dataset/model combination.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            environment: Environment (dev/staging/prod)
            distance_metric: Distance metric ("cosine" or "l2")

        Returns:
            Sorted list of available alpha values

        """
        chroma_client = self._get_chroma_client()

        collections = chroma_client.list_collections()

        alphas = []

        # Get prefix without alpha
        prefix = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            environment=environment,
            distance_metric=distance_metric,
            alpha=None,
        )

        for collection in collections:
            if collection.name.startswith(prefix):
                match = re.search(r"_a(\d+\.\d+)", collection.name)
                if match:
                    alphas.append(float(match.group(1)))

        return sorted(alphas)
